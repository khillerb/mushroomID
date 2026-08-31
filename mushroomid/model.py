"""Transfer-learning classifier built on an ImageNet backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models

# name -> (constructor, default weights enum, attribute holding the classifier head)
BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.DEFAULT,
        "classifier",
    ),
}

DEFAULT_BACKBONE = "resnet50"


@dataclass
class ModelSpec:
    """Serialisable description of a trained model, stored in checkpoints."""

    backbone: str
    num_classes: int
    classes: list[str]
    image_size: int


def _replace_head(model: nn.Module, head_attr: str, num_classes: int, dropout: float) -> None:
    head = getattr(model, head_attr)

    if isinstance(head, nn.Sequential):
        # EfficientNet ships Dropout + Linear; keep the shape, swap the Linear.
        in_features = head[-1].in_features
        setattr(
            model,
            head_attr,
            nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes)),
        )
    else:
        in_features = head.in_features
        setattr(
            model,
            head_attr,
            nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes)),
        )


def build_model(
    num_classes: int,
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Return a backbone with a fresh classification head.

    With `freeze_backbone=True` only the head trains -- fast, and the sensible
    first pass on ~6.7k images. Unfreeze for a low-learning-rate fine-tune once
    the head has converged.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from {sorted(BACKBONES)}")

    constructor, weights, head_attr = BACKBONES[backbone]
    model = constructor(weights=weights if pretrained else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # The new head is created after freezing, so it always stays trainable.
    _replace_head(model, head_attr, num_classes, dropout)
    return model


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unfreeze_backbone(model: nn.Module) -> None:
    """Make every parameter trainable, for a fine-tuning pass."""
    for param in model.parameters():
        param.requires_grad = True


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[nn.Module, ModelSpec]:
    """Rebuild a model from a checkpoint written by `train.py`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    spec = ModelSpec(**payload["spec"])
    model = build_model(
        num_classes=spec.num_classes,
        backbone=spec.backbone,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, spec
