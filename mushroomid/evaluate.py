"""Evaluate a trained checkpoint on the held-out test split.

Example:
    python -m mushroomid.evaluate --checkpoint outputs/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .config import BATCH_SIZE, IMAGE_DIR, OUTPUT_DIR, SEED
from .data import build_dataloaders
from .model import load_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "best_model.pt")
    parser.add_argument("--data-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="which partition to score",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip writing the confusion-matrix image",
    )
    return parser.parse_args(argv)


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in tqdm(loader, desc="scoring", leave=False):
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(targets.tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    """Write a row-normalised confusion matrix image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted genus")
    ax.set_ylabel("true genus")
    ax.set_title("Confusion matrix (row-normalised)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{normalised[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalised[i, j] > 0.5 else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}. Run `python -m mushroomid.train` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spec = load_checkpoint(args.checkpoint, device)

    data = build_dataloaders(
        image_dir=args.data_dir,
        image_size=spec.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    if data.classes != spec.classes:
        raise SystemExit(
            "Class list in the checkpoint does not match the data directory.\n"
            f"  checkpoint: {spec.classes}\n"
            f"  data dir  : {data.classes}"
        )

    loader = data.test if args.split == "test" else data.val
    y_true, y_pred = collect_predictions(model, loader, device)

    accuracy = float((y_true == y_pred).mean())
    print(f"\n{args.split} accuracy: {accuracy:.4f}  ({len(y_true)} images)\n")
    print(
        classification_report(
            y_true, y_pred, target_names=spec.classes, digits=3, zero_division=0
        )
    )

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(spec.classes)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / f"confusion_matrix_{args.split}.csv"
    np.savetxt(
        csv_path,
        matrix,
        fmt="%d",
        delimiter=",",
        header=",".join(spec.classes),
        comments="",
    )
    print(f"confusion matrix : {csv_path}")

    if not args.no_plot:
        plot_path = args.output_dir / f"confusion_matrix_{args.split}.png"
        plot_confusion_matrix(matrix, spec.classes, plot_path)
        print(f"plot             : {plot_path}")


if __name__ == "__main__":
    main()
