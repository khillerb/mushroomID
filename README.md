# mushroomID

Transfer-learning image classifier that predicts the **genus** of a mushroom
from a photograph, across nine common genera. Built on PyTorch and torchvision,
with a stratified data pipeline, a training CLI, evaluation reporting, and
batch inference.

---

> ### ⚠️ Safety notice
>
> **This is a machine-learning exercise, not a foraging tool.** It predicts
> genus, not species, and it is wrong a meaningful fraction of the time.
>
> Genus is not sufficient to determine edibility. Several genera in this
> dataset — *Amanita* and *Cortinarius* above all — contain species that are
> lethal in small quantities, and those species look similar to edible ones in
> photographs. Deaths occur every year from exactly this kind of mistake.
>
> **Never use this model, or any image classifier, to decide whether a mushroom
> is safe to eat.** Consult an experienced local mycologist and a regional field
> guide.

---

## Overview

The classifier fine-tunes an ImageNet-pretrained backbone (ResNet-50 by
default) on ~6,700 photographs spanning nine genera. The default recipe freezes
the backbone and trains only a fresh classification head, which converges
quickly on a dataset this size; passing `--unfreeze-at` switches to a two-stage
schedule that unfreezes the backbone partway through at a 10x lower learning
rate.

Two details of the data drive most of the design:

- **Class imbalance is substantial** — *Lactarius* has 1,563 images to
  *Suillus*'s 311, a 5x spread. The loss is weighted by inverse class frequency
  by default, and splits are stratified per class so every genus is represented
  in train, validation, and test.
- **Photographs vary wildly** in framing, lighting, and orientation. Geometric
  augmentation is aggressive; colour augmentation is deliberately mild, because
  cap and gill colour carry real taxonomic signal.

## Project structure

```
mushroomID/
├── mushroomid/
│   ├── config.py       Paths, genus list, normalisation constants, defaults
│   ├── data.py         ImageFolder loading, stratified splits, augmentation
│   ├── model.py        Backbone selection, head replacement, checkpoint I/O
│   ├── train.py        Training CLI with checkpointing and history logging
│   ├── evaluate.py     Per-class report and confusion matrix on a held-out split
│   ├── predict.py      Batch inference over files or directories
│   └── scrape.py       Species reference scraper (Wild Food UK)
├── data/
│   ├── mushrooms/      Image set, one directory per genus (not in git)
│   └── species.csv     Genus reference table
├── outputs/            Checkpoints, history, evaluation artifacts (not in git)
├── requirements.txt
└── pyproject.toml
```

## Setup

Create and activate a virtual environment — `.venv\Scripts\activate` on
Windows, `source .venv/bin/activate` elsewhere:

```bash
python -m venv .venv
```

```bash
pip install -r requirements.txt
```

The pinned `torch` and `torchvision` in `requirements.txt` install the CPU build
by default. For CUDA, install them from the PyTorch index first, then install
the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Dataset

The image set is **not tracked in git** — it is roughly 1.9 GB. Arrange it as
one directory per genus:

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

6,714 images total, split 70/15/15 into train/validation/test. The split is
stratified and seeded (`--seed`, default 42), so it is reproducible across runs.

`data/species.csv` records the common name, image count, and a short
identification note for each genus. It is reference material for interpreting
predictions, **not** an edibility guide.

## Usage

Train with the default recipe — frozen ResNet-50 backbone, 15 epochs:

```bash
python -m mushroomid.train
```

Two-stage fine-tune, unfreezing the backbone at epoch 5:

```bash
python -m mushroomid.train --epochs 20 --unfreeze-at 5 --backbone resnet50
```

Score the held-out test split and write a confusion matrix:

```bash
python -m mushroomid.evaluate --checkpoint outputs/best_model.pt
```

Classify new photographs:

```bash
python -m mushroomid.predict photo.jpg --top-k 3
```

```bash
python -m mushroomid.predict ./finds/ --json
```

Refresh the species reference table from the Wild Food UK guide:

```bash
python -m mushroomid.scrape --output data/species_reference.csv --delay 1.0
```

### Useful flags

| Flag | Applies to | Purpose |
| --- | --- | --- |
| `--backbone` | train | `resnet18`, `resnet50`, or `efficientnet_b0` |
| `--unfreeze-at N` | train | Unfreeze the backbone at epoch N, LR ÷ 10 |
| `--no-class-weights` | train | Disable inverse-frequency loss weighting |
| `--no-pretrained` | train | Random init, for offline runs and ablations |
| `--limit-batches N` | train | Stop each epoch after N batches (smoke test) |
| `--num-workers` | train, evaluate | Defaults to 0; raise on Linux for a speedup |
| `--split val` | evaluate | Score validation instead of test |
| `--json` | predict | Machine-readable output |

Training writes to `outputs/`: `best_model.pt` (best validation accuracy, with
the class list and image size embedded), `history.csv` (per-epoch losses,
accuracies, and learning rate), and `train_args.json`.

## Results

No trained checkpoint is committed to this repository — the weights and the
image set are both excluded from git. Run `python -m mushroomid.train` followed
by `python -m mushroomid.evaluate` to produce a per-class precision/recall
report and a confusion matrix under `outputs/`.

Expect genus pairs that are hard for humans to be hard here too, particularly
*Russula* against *Lactarius*, and brown-capped *Cortinarius* against
*Agaricus*.

## Limitations

- **Genus only.** Species-level identification is the part that matters for
  edibility, and this model does not attempt it.
- **No "unknown" class.** Every image is forced into one of nine genera. A
  photograph of a tenth genus, or of something that is not a mushroom at all,
  still produces a confident-looking prediction.
- **Web-scraped labels.** The images were collected by genus folder without
  expert verification; some are likely mislabelled, and the reported accuracy
  ceiling reflects that.
- **Photographic features only.** Spore print, smell, latex colour, substrate,
  and habitat carry much of the real diagnostic signal and are absent from a
  single image.

## License

Not yet chosen. Note that the image set was scraped from third-party sources
and its licensing has not been established — resolve that before redistributing
`data/mushrooms/`.
