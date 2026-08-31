"""Dataset discovery, splitting, and augmentation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_DIR,
    SEED,
    TEST_SPLIT,
    VAL_SPLIT,
)

# A handful of images in the scraped set are truncated; decode them anyway
# rather than failing an entire epoch.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmentation pipeline for training.

    Mushroom photographs vary wildly in framing, lighting, and orientation, so
    the augmentations are deliberately aggressive on geometry and mild on
    colour -- cap and gill colour carry real taxonomic signal.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic pipeline for validation, test, and inference."""
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@dataclass
class DataBundle:
    """Everything training needs to know about the data."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    classes: list[str]
    class_weights: torch.Tensor
    counts: dict[str, int]

    def describe(self) -> str:
        lines = [
            f"classes      : {len(self.classes)}",
            f"train images : {len(self.train.dataset)}",
            f"val images   : {len(self.val.dataset)}",
            f"test images  : {len(self.test.dataset)}",
            "per-class counts:",
        ]
        width = max(len(name) for name in self.classes)
        for name in self.classes:
            lines.append(f"  {name:<{width}}  {self.counts[name]:>5}")
        return "\n".join(lines)


def stratified_split(
    targets: list[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices per class so every genus appears in every partition.

    A plain random split would leave the smallest genera (Suillus, Hygrocybe)
    badly represented in the test set.
    """
    if not 0 < val_split + test_split < 1:
        raise ValueError("val_split + test_split must be between 0 and 1")

    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    train_idx, val_idx, test_idx = [], [], []

    for class_id in np.unique(targets_arr):
        idx = np.flatnonzero(targets_arr == class_id)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_split)))
        n_test = max(1, int(round(len(idx) * test_split)))
        val_idx.append(idx[:n_val])
        test_idx.append(idx[n_val : n_val + n_test])
        train_idx.append(idx[n_val + n_test :])

    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def build_dataloaders(
    image_dir: Path = IMAGE_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = SEED,
    num_workers: int = 0,
) -> DataBundle:
    """Load the ImageFolder tree and return stratified train/val/test loaders.

    `num_workers` defaults to 0 because worker processes on Windows re-import
    the entry module; raise it on Linux for a healthy speedup.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"No image directory at {image_dir}. See the Dataset section of README.md."
        )

    # Two views of the same tree so train and eval get different transforms.
    train_view = datasets.ImageFolder(image_dir, transform=train_transform(image_size))
    eval_view = datasets.ImageFolder(image_dir, transform=eval_transform(image_size))

    if not train_view.classes:
        raise RuntimeError(f"{image_dir} contains no class subdirectories")

    train_idx, val_idx, test_idx = stratified_split(
        train_view.targets, val_split, test_split, seed
    )

    counts = Counter(train_view.targets)
    ordered_counts = {name: counts[i] for i, name in enumerate(train_view.classes)}

    # Inverse-frequency weights: Lactarius has ~5x the images of Suillus.
    frequencies = torch.tensor(
        [counts[i] for i in range(len(train_view.classes))], dtype=torch.float
    )
    class_weights = frequencies.sum() / (len(frequencies) * frequencies)

    generator = torch.Generator().manual_seed(seed)

    def loader(view, indices, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(view, indices.tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if shuffle else None,
        )

    return DataBundle(
        train=loader(train_view, train_idx, shuffle=True),
        val=loader(eval_view, val_idx, shuffle=False),
        test=loader(eval_view, test_idx, shuffle=False),
        classes=list(train_view.classes),
        class_weights=class_weights,
        counts=ordered_counts,
    )
