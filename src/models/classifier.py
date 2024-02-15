"""Wafer defect pattern classifier — transfer learning with configurable backbone."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


BACKBONE_MAP = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT, 1536),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, 768),
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
        in_channels: int = 3,
    ):
        super().__init__()
        factory, weights, feat_dim = BACKBONE_MAP[backbone_name]
        backbone = factory(weights=weights if pretrained else None)

        # Adapt first conv for configurable input channels
        if in_channels != 3 and "resnet" in backbone_name:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    if in_channels > 3:
                        nn.init.kaiming_normal_(new_conv.weight[:, 3:])
            backbone.conv1 = new_conv

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


class WaferClassifierFPN(nn.Module):
    """ResNet50 + Feature Pyramid Network neck for multi-scale classification.

    Attaches an FPN to ResNet50's C2–C5 feature maps, producing P2–P5 with
    fpn_out_channels each. Each FPN level is globally average-pooled
    independently, then concatenated and fed through the classifier head.

    This simultaneously captures:
    - P2/C2: fine-grained Scratch details and small Loc clusters (~8×8 receptive)
    - P3/C3: medium-scale patterns (~16×16 receptive)
    - P4/C4: large Edge-Ring / Edge-Loc patterns (~32×32 receptive)
    - P5/C5: whole-wafer global patterns (Center, Donut, Near-Full)

    Supports configurable input channels for the engineered-feature pipeline
    (in_channels=5: die_mask, pass, fail, radon, dist_center).

    Args:
        num_classes: number of defect classes
        in_channels: number of input channels (3 for standard, 5 for +engineered)
        pretrained: initialise backbone from ImageNet weights
        dropout: dropout rate before the dense layers
        fpn_out_channels: channels per FPN level (256 matches original FPN paper)
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        fpn_out_channels: int = 256,
    ):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        # Adapt first conv for configurable input channels
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    if in_channels > 3:
                        nn.init.kaiming_normal_(new_conv.weight[:, 3:])
            backbone.conv1 = new_conv

        # Decompose backbone into stages for FPN attachment
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # C2: 256 ch, stride 4
        self.layer2 = backbone.layer2   # C3: 512 ch, stride 8
        self.layer3 = backbone.layer3   # C4: 1024 ch, stride 16
        self.layer4 = backbone.layer4   # C5: 2048 ch, stride 32

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Head: 4 FPN levels × fpn_out_channels → 512 → num_classes
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * fpn_out_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def _extract_pyramid(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn(OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fpn_out = self._extract_pyramid(x)
        pooled = [self.pool(fpn_out[k]) for k in ["c2", "c3", "c4", "c5"]]
        feat = torch.cat(pooled, dim=1)
        return self.head(feat)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened FPN feature vector (for analysis / t-SNE)."""
        fpn_out = self._extract_pyramid(x)
        pooled = [self.pool(fpn_out[k]) for k in ["c2", "c3", "c4", "c5"]]
        return torch.cat(pooled, dim=1).flatten(1)


def build_classifier(cfg) -> nn.Module:
    """Factory that selects WaferClassifier or WaferClassifierFPN from config."""
    backbone = cfg.classifier.backbone
    in_channels = 3 + sum([
        cfg.get("features", {}).get("use_radon", False),
        cfg.get("features", {}).get("use_distance_from_center", False),
    ])

    if backbone == "resnet50_fpn":
        return WaferClassifierFPN(
            num_classes=cfg.wm811k.num_classes,
            in_channels=in_channels,
            pretrained=cfg.classifier.pretrained,
            fpn_out_channels=cfg.classifier.get("fpn_out_channels", 256),
        )
    return WaferClassifier(
        backbone_name=backbone,
        num_classes=cfg.wm811k.num_classes,
        pretrained=cfg.classifier.pretrained,
        in_channels=in_channels,
    )
