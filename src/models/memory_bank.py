"""Coreset-subsampled memory bank for PatchCore.

Stores representative patch-level features from the training set.
Uses greedy coreset selection to reduce memory while maintaining coverage.
"""

from __future__ import annotations

import torch
import numpy as np
from src.utils.logging import get_logger

log = get_logger(__name__)


class MemoryBank:
    """Memory bank with coreset subsampling for efficient anomaly scoring."""

    def __init__(self, coreset_ratio: float = 0.01):
        self.coreset_ratio = coreset_ratio
        self.features: torch.Tensor | None = None

    def fit(self, all_features: torch.Tensor, device: str = "auto") -> None:
        """Build memory bank from training patch features.

        Args:
            all_features: (N, C) tensor of all patch embeddings from normal data
            device: device to run coreset selection on ("auto" uses CUDA if available)
        """
        log.info("building_memory_bank", total_patches=all_features.shape[0])

        n_select = max(1, int(all_features.shape[0] * self.coreset_ratio))
        selected_idx = self._greedy_coreset(all_features, n_select, device)
        self.features = all_features[selected_idx]

        log.info("memory_bank_ready", bank_size=self.features.shape[0])

    def _greedy_coreset(
        self, features: torch.Tensor, n_select: int, device: str = "auto"
    ) -> list[int]:
        """Greedy coreset selection (farthest-point sampling) on GPU.

        Uses chunked torch.cdist so the full N×N matrix is never materialised.
        Each iteration only computes distances from all points to the latest
        selected point (one column), keeping memory O(N) not O(N²).
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        f = features.to(device)
        N = f.shape[0]
        chunk = 8192  # rows to process at once — tune to fit VRAM

        selected: list[int] = [int(torch.randint(N, (1,)).item())]
        # min_dists[i] = distance from point i to nearest selected point so far
        min_dists = torch.full((N,), float("inf"), device=device)

        def _update_min_dists(new_idx: int) -> None:
            anchor = f[new_idx].unsqueeze(0)  # (1, C)
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                d = torch.cdist(f[start:end], anchor).squeeze(1)
                min_dists[start:end] = torch.minimum(min_dists[start:end], d)

        _update_min_dists(selected[0])

        for _ in range(n_select - 1):
            idx = int(min_dists.argmax().item())
            selected.append(idx)
            _update_min_dists(idx)

        return selected

    def score(self, query_features: torch.Tensor, k: int = 9) -> torch.Tensor:
        """Compute anomaly scores via k-nearest-neighbor distance.

        Args:
            query_features: (M, C) patch embeddings from test image
            k: number of neighbors

        Returns:
            scores: (M,) anomaly score per patch
        """
        dists = torch.cdist(query_features, self.features)
        topk_dists, _ = dists.topk(k, largest=False, dim=1)
        return topk_dists.mean(dim=1)

    def save(self, path: str) -> None:
        torch.save({"features": self.features, "coreset_ratio": self.coreset_ratio}, path)

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        data = torch.load(path, weights_only=True)
        bank = cls(coreset_ratio=data["coreset_ratio"])
        bank.features = data["features"]
        return bank
