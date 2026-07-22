"""Supervised classification of segmented 3D objects via a small CNN.

Replaces vtea.deeplearning's native Java 3D CNN stack (Generic3DCNN,
NephNet3D, TensorConverter, Trainer - wired up via JavaCPP PyTorch
bindings) with plain PyTorch: no binding layer needed, and the
train/predict loop is ordinary PyTorch boilerplate rather than a bespoke
Java training harness. This is a small, generic architecture, not a
reimplementation of NephNet3D's specific layer choices - swap in a
different torch.nn.Module for a different architecture as needed.

Requires the `deeplearning` extra (torch). Unlike
vtea_core.segmentation.deep (where cellpose is only imported inside a
function), this module imports torch at module load time, since defining
an nn.Module subclass requires it - importing vtea_core.classification at
all requires the extra to be installed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Cell3DClassifier(nn.Module):
    """A small 3D CNN classifier for fixed-size (D, H, W) single-channel object crops."""

    def __init__(self, n_classes: int, *, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features)


def train_classifier(
    model: nn.Module,
    crops: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """Trains `model` in place on (N, D, H, W) crops with integer class `labels`.

    Plain supervised training (cross-entropy loss, Adam optimizer) - no
    VTEA-specific machinery, since that's exactly the kind of boilerplate
    native PyTorch replaces the Java stack's Trainer/TensorConverter
    classes for.
    """
    model.to(device).train()
    x = torch.as_tensor(crops, dtype=torch.float32, device=device).unsqueeze(1)  # add channel dim
    y = torch.as_tensor(labels, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    return model


def predict(model: nn.Module, crops: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """Predicted class ids for (N, D, H, W) crops."""
    model.to(device).eval()
    x = torch.as_tensor(crops, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(dim=1).cpu().numpy()
