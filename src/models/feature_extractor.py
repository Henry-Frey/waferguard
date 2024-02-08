"""Pretrained feature extractor for PatchCore anomaly detection.

Extracts intermediate feature maps from a pretrained backbone at specified
layers. These patch-level features form the basis of the memory bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """Hook-based feature extraction from pretrained backbone layers."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: list[str] = ("layer2", "layer3"),
    ):
        super().__init__()
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            self.backbone = models.wide_resnet50_2(weights=weights)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.layers = layers
        self._features: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in self.layers:
            layer = dict(self.backbone.named_children())[name]
            layer.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, name: str):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._features = {}
        self.backbone(x)
        return {k: v.clone() for k, v in self._features.items()}

    def get_patch_features(self, x: torch.Tensor, target_size: int = 28) -> torch.Tensor:
        """Extract and concatenate multi-scale patch features.

        All feature maps are resized to target_size and concatenated along
        the channel dimension, then reshaped to (B*H*W, C) patch embeddings.
        """
        feats = self.forward(x)
        resized = []
        for name in self.layers:
            f = feats[name]
            if f.shape[2] != target_size:
                f = nn.functional.interpolate(
                    f, size=target_size, mode="bilinear", align_corners=False
                )
            resized.append(f)

        concat = torch.cat(resized, dim=1)  # (B, C_total, H, W)
        B, C, H, W = concat.shape
        return concat.permute(0, 2, 3, 1).reshape(B * H * W, C)
