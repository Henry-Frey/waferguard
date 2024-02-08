"""PatchCore anomaly detection — complete inference pipeline.

PatchCore (Roth et al., 2022) achieves SOTA anomaly detection by:
1. Extracting patch-level features from a pretrained backbone
2. Building a memory bank of normal patch features (coreset-subsampled)
3. Scoring test patches by distance to nearest memory bank neighbors
4. Generating pixel-level anomaly heatmaps

This is ideal for semiconductor inspection where defective samples
are extremely rare and defect types are unknown at training time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.feature_extractor import FeatureExtractor
from src.models.memory_bank import MemoryBank
from src.utils.logging import get_logger

log = get_logger(__name__)


class PatchCore:
    """PatchCore anomaly detection pipeline."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: list[str] = ("layer2", "layer3"),
        coreset_ratio: float = 0.01,
        k_nearest: int = 9,
        device: str = "auto",
    ):
        self.device = torch.device(
            device if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.extractor = FeatureExtractor(backbone_name, layers).to(self.device)
        self.memory_bank = MemoryBank(coreset_ratio=coreset_ratio)
        self.k = k_nearest
        self._patch_size: int | None = None

    @torch.no_grad()
    def fit(self, train_loader) -> None:
        """Build memory bank from normal (defect-free) training data."""
        log.info("patchcore_fitting")
        self.extractor.eval()
        all_features = []

        for batch in train_loader:
            images = batch["image"].to(self.device)
            patches = self.extractor.get_patch_features(images)
            all_features.append(patches.cpu())

        all_features = torch.cat(all_features, dim=0)
        self.memory_bank.fit(all_features, device=str(self.device))

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run anomaly detection on a batch of images.

        Returns:
            image_scores: (B,) anomaly score per image
            anomaly_maps: (B, H, W) pixel-level anomaly heatmaps
        """
        self.extractor.eval()
        B = images.shape[0]
        patches = self.extractor.get_patch_features(images.to(self.device))

        # score each patch
        scores = self.memory_bank.score(patches, k=self.k)

        # reshape to spatial map
        side = int(np.sqrt(patches.shape[0] // B))
        maps = scores.reshape(B, side, side)

        # upsample to input resolution
        anomaly_maps = F.interpolate(
            maps.unsqueeze(1),
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # image-level score = max patch score
        image_scores = anomaly_maps.flatten(1).max(dim=1).values

        return image_scores.cpu(), anomaly_maps.cpu()

    def save(self, path: str) -> None:
        self.memory_bank.save(path)

    def load(self, path: str) -> None:
        self.memory_bank = MemoryBank.load(path)
