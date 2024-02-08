"""Loss functions for imbalanced wafer defect classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for handling class imbalance.

    In WM-811K, class distribution is heavily skewed:
    Edge-Ring ~9680 samples vs Near-Full ~149 samples.
    Focal loss down-weights well-classified examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 9):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing to improve generalization."""

    def __init__(self, num_classes: int = 9, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full_like(pred, smooth_val)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        log_prob = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prob).sum(dim=1).mean()
