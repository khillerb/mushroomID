"""Train the mushroom genus classifier.

Example:
    python -m mushroomid.train --epochs 15 --backbone resnet50
    python -m mushroomid.train --epochs 8 --unfreeze-at 4 --lr 3e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .config import BATCH_SIZE, IMAGE_DIR, IMAGE_SIZE, OUTPUT_DIR, SEED
from .data import build_dataloaders
from .model import DEFAULT_BACKBONE, build_model, trainable_parameters, unfreeze_backbone


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mushroom genus classifier.")
    parser.add_argument("--data-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--unfreeze-at",
        type=int,
        default=None,
        metavar="EPOCH",
        help="unfreeze the backbone at this epoch and drop the LR 10x",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="start from random weights (offline / ablation)",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="disable inverse-frequency loss weighting",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="stop each epoch after N batches (smoke test)",
    )
    return parser.parse_args(argv)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    limit_batches: int | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """Run one pass. Trains when `optimizer` is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0
    use_amp = scaler is not None and device.type == "cuda"

    progress = tqdm(loader, desc=desc, leave=False, total=limit_batches or len(loader))
    with torch.set_grad_enabled(training):
        for batch_num, (images, targets) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)
            progress.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.3f}")

            if limit_batches and batch_num >= limit_batches:
                break

    progress.close()
    return total_loss / max(seen, 1), correct / max(seen, 1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataloaders(
        image_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(data.describe())

    model = build_model(
        num_classes=len(data.classes),
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    ).to(device)

    print(f"\ndevice={device}  backbone={args.backbone}")
    print(f"trainable parameters: {trainable_parameters(model):,}\n")

    weights = None if args.no_class_weights else data.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    history_path = args.output_dir / "history.csv"
    checkpoint_path = args.output_dir / "best_model.pt"
    best_val_acc = 0.0
    started = time.time()

    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

        for epoch in range(1, args.epochs + 1):
            if args.unfreeze_at and epoch == args.unfreeze_at:
                print(f"epoch {epoch}: unfreezing backbone, LR -> {args.lr / 10:g}")
                unfreeze_backbone(model)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr / 10, weight_decay=args.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - epoch + 1)
                )

            train_loss, train_acc = run_epoch(
                model,
                data.train,
                criterion,
                device,
                optimizer,
                scaler,
                args.limit_batches,
                f"epoch {epoch}/{args.epochs} train",
            )
            val_loss, val_acc = run_epoch(
                model,
                data.val,
                criterion,
                device,
                limit_batches=args.limit_batches,
                desc=f"epoch {epoch}/{args.epochs} val",
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch {epoch:>3}/{args.epochs}  "
                f"train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{current_lr:.8f}",
                ]
            )
            fh.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "spec": {
                            "backbone": args.backbone,
                            "num_classes": len(data.classes),
                            "classes": data.classes,
                            "image_size": args.image_size,
                        },
                        "val_acc": val_acc,
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
                print(f"           saved checkpoint (val acc {val_acc:.4f})")

    elapsed = time.time() - started
    print(f"\nfinished in {elapsed / 60:.1f} min -- best val accuracy {best_val_acc:.4f}")
    print(f"checkpoint : {checkpoint_path}")
    print(f"history    : {history_path}")

    (args.output_dir / "train_args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
