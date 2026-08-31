"""Project-wide paths and defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "mushrooms"
SPECIES_CSV = DATA_DIR / "species.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# The nine genera present in the image set.
GENERA = (
    "Agaricus",
    "Amanita",
    "Boletus",
    "Cortinarius",
    "Entoloma",
    "Hygrocybe",
    "Lactarius",
    "Russula",
    "Suillus",
)

# ImageNet normalisation -- all supported backbones are ImageNet-pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

SAFETY_NOTICE = (
    "This model predicts genus from a photograph and is frequently wrong. "
    "Never use it to decide whether a mushroom is safe to eat."
)
