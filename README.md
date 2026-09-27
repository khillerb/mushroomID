# mushroomID

This project looks at a photo of a mushroom and predicts its **genus**. It
covers nine common genera. It's built with PyTorch and torchvision and uses
transfer learning. That means it starts from a model that already knows a lot
about images, then teaches it this new task. The project includes a data
pipeline, a training command, an evaluation report, and batch predictions.

---

> ### ⚠️ Safety notice
>
> **This is a machine learning exercise, not a foraging tool.** It only
> predicts the genus, not the species, and it's wrong a fair amount of the
> time.
>
> Knowing the genus isn't enough to tell if a mushroom is safe to eat.
> Several genera in this dataset, especially *Amanita* and *Cortinarius*,
> include species that can kill you in small amounts. Those species can look
> a lot like edible ones in photos, and people die every year from exactly
> this kind of mistake.
>
> **Never use this model, or any image classifier, to decide whether a
> mushroom is safe to eat.** Ask an experienced local mycologist and use a
> field guide for your region.

---

## Overview

The classifier fine-tunes a backbone that was pretrained on ImageNet (ResNet-50
by default). It trains on about 6,700 photos across nine genera. By default,
the backbone stays frozen and only a new classification head is trained. That
learns quickly on a dataset this size. If you pass `--unfreeze-at`, training
switches to two stages: partway through, the backbone is unfrozen and the
learning rate drops by 10x so the pretrained weights only shift a little.

Two things about the data shape most of the design:

- **Some genera have far more photos than others.** *Lactarius* has 1,563
  images, while *Suillus* has only 311, a 5x gap. So by default, the loss gives
  rarer genera more weight. The splits are also stratified per class, which
  means every genus shows up in train, validation, and test.
- **The photos are very different from each other** in framing, lighting,
  and angle. Because of that, the geometric augmentation (crops, flips,
  rotations) is strong. The color augmentation is kept light on purpose,
  though, because cap and gill color really help tell genera apart.

## Project structure

```
mushroomID/
├── mushroomid/
│   ├── config.py       Paths, genus list, normalization values, defaults
│   ├── data.py         Image loading, stratified splits, augmentation
│   ├── model.py        Backbone choice, new head, checkpoint loading
│   ├── train.py        Training command with checkpoints and history logs
│   ├── evaluate.py     Per-class report and confusion matrix on held-out data
│   ├── predict.py      Batch predictions over files or folders
│   └── scrape.py       Species reference scraper (Wild Food UK)
├── data/
│   ├── mushrooms/      Image set, one folder per genus (not in git)
│   └── species.csv     Genus reference table
├── outputs/            Checkpoints, history, evaluation files (not in git)
├── requirements.txt
└── pyproject.toml
```

## Setup

First, create a virtual environment:

```bash
python -m venv .venv
```

Then activate it (`.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` everywhere else) and install the requirements:

```bash
pip install -r requirements.txt
```

By default, `requirements.txt` installs the CPU build of `torch` and
`torchvision`. If you want to train on a GPU, install those two from the
PyTorch CUDA index first, and then install the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Dataset

The image set is **not in git**, because it's about 1.9 GB. It should be set
up with one folder per genus:

```
data/mushrooms/
├── Agaricus/      353 images
├── Amanita/       750
├── Boletus/     1,073
├── Cortinarius/   836
├── Entoloma/      364
├── Hygrocybe/     316
├── Lactarius/   1,563
├── Russula/     1,148
└── Suillus/       311
```

That's 6,714 images in total, split 70/15/15 into train, validation, and
test. The split is stratified and uses a fixed seed (`--seed`, default 42), so
you get the same split every run.

`data/species.csv` lists each genus with its common name, image count, and a
short ID note. It's there to help you read the predictions. It is **not** an
edibility guide.

## Usage

To train with the default setup (frozen ResNet-50 backbone, 15 epochs):

```bash
python -m mushroomid.train
```

To fine-tune in two stages and unfreeze the backbone at epoch 5:

```bash
python -m mushroomid.train --epochs 20 --unfreeze-at 5 --backbone resnet50
```

To score the held-out test set and save a confusion matrix:

```bash
python -m mushroomid.evaluate --checkpoint outputs/best_model.pt
```

To classify new photos:

```bash
python -m mushroomid.predict photo.jpg --top-k 3
```

```bash
python -m mushroomid.predict ./finds/ --json
```

To refresh the species reference table from the Wild Food UK guide:

```bash
python -m mushroomid.scrape --output data/species_reference.csv --delay 1.0
```

### Useful flags

| Flag | Used by | What it does |
| --- | --- | --- |
| `--backbone` | train | Pick `resnet18`, `resnet50`, or `efficientnet_b0` |
| `--unfreeze-at N` | train | Unfreeze the backbone at epoch N and divide the learning rate by 10 |
| `--no-class-weights` | train | Turn off the extra loss weight for rare genera |
| `--no-pretrained` | train | Start from random weights, for offline runs or comparisons |
| `--limit-batches N` | train | Stop each epoch after N batches, for a quick test run |
| `--num-workers` | train, evaluate | Defaults to 0; raise it on Linux for faster loading |
| `--split val` | evaluate | Score the validation set instead of the test set |
| `--json` | predict | Print results as JSON |

Training saves its results to `outputs/`:

- `best_model.pt`: the checkpoint with the best validation accuracy, with the
  class list and image size saved inside.
- `history.csv`: loss, accuracy, and learning rate for each epoch.
- `train_args.json`: the settings used for the run.

## Results

This repository doesn't include a trained checkpoint, because the weights and
the image set are both left out of git. To get results, run
`python -m mushroomid.train` and then `python -m mushroomid.evaluate`. That
writes a per-class precision and recall report and a confusion matrix to
`outputs/`.

Genera that are hard for people to tell apart will probably be hard for the
model too. The main ones to watch are *Russula* vs. *Lactarius*, and
brown-capped *Cortinarius* vs. *Agaricus*.

## Limitations

- **Genus only.** Edibility depends on the exact species, and this model
  doesn't try to go that far.
- **No "unknown" class.** Every photo gets forced into one of the nine
  genera. A photo of a tenth genus, or of something that isn't a mushroom at
  all, still gets a prediction that can look confident.
- **Labels come from the web.** The images were collected by genus folder
  without an expert checking them. Some are probably mislabeled, and that
  limits how high the accuracy can go.
- **Photos only.** A lot of real ID information, like spore print, smell,
  latex color, what the mushroom grows on, and habitat, doesn't show up in a
  single photo.

## License

A license hasn't been chosen yet. Also, the image set was scraped from
third-party sites and its licensing isn't clear, so that needs to be sorted
out before anyone shares `data/mushrooms/`.
