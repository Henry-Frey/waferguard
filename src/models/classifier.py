"""Wafer defect pattern classifier — transfer learning with configurable backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


BACKBONE_MAP = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
}


class WaferClassifier(nn.Module):
    """Transfer-learning classifier for wafer defect patterns.

    Architecture:
        Pretrained backbone → Global Average Pool → Dropout → FC → Classes

    The first conv layer is adapted from 3-channel RGB to our 3-channel
    wafer encoding (die_mask, pass, fail).
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        num_classes: int = 9,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        factory, weights, feat_dim = BACKBONE_MAP[backbone_name]
        backbone = factory(weights=weights if pretrained else None)

        # for ResNet: remove final FC, keep everything else
        if "resnet" in backbone_name:
            self.features = nn.Sequential(*list(backbone.children())[:-1])
        else:
            self.features = backbone.features
            feat_dim = backbone.classifier[-1].in_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        return self.head(feat)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened feature vector (for analysis / t-SNE)."""
        feat = self.features(x)
        return self.pool(feat).flatten(1)
