"""Run a trained checkpoint over one or more images.

Example:
    python -m mushroomid.predict photo.jpg
    python -m mushroomid.predict ./finds/ --top-k 3 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from .config import OUTPUT_DIR, SAFETY_NOTICE
from .data import eval_transform
from .model import load_checkpoint

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify mushroom photographs by genus.")
    parser.add_argument("paths", type=Path, nargs="+", help="image files or directories")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "best_model.pt")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return parser.parse_args(argv)


def gather_images(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif path.is_file():
            found.append(path)
        else:
            print(f"warning: no such path {path}", file=sys.stderr)
    return found


@torch.no_grad()
def predict_one(model, transform, image_path: Path, classes: list[str], device, top_k: int):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    probabilities = model(tensor).softmax(dim=1).squeeze(0)
    scores, indices = probabilities.topk(min(top_k, len(classes)))
    return [
        {"genus": classes[i], "probability": round(float(s), 4)}
        for s, i in zip(scores.tolist(), indices.tolist(), strict=True)
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {args.checkpoint}. Run `python -m mushroomid.train` first."
        )

    images = gather_images(args.paths)
    if not images:
        raise SystemExit("No images found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spec = load_checkpoint(args.checkpoint, device)
    transform = eval_transform(spec.image_size)

    results = []
    for image_path in images:
        try:
            predictions = predict_one(model, transform, image_path, spec.classes, device, args.top_k)
        except (UnidentifiedImageError, OSError) as exc:
            print(f"warning: could not read {image_path}: {exc}", file=sys.stderr)
            continue
        results.append({"path": str(image_path), "predictions": predictions})

    if args.json:
        print(json.dumps({"notice": SAFETY_NOTICE, "results": results}, indent=2))
        return

    for result in results:
        print(f"\n{result['path']}")
        for rank, prediction in enumerate(result["predictions"], start=1):
            bar = "#" * int(prediction["probability"] * 40)
            print(f"  {rank}. {prediction['genus']:<12} {prediction['probability']:>6.1%}  {bar}")

    print(f"\n{SAFETY_NOTICE}")


if __name__ == "__main__":
    main()
