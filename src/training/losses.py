"""Loss functions for imbalanced wafer defect classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with optional per-class alpha weights.

    For WM-811K the class distribution spans 4 orders of magnitude:
      None ~147k, Near-Full ~149.  A scalar alpha cannot handle this;
      per-class alpha (inverse-frequency) is the correct formulation.

    Args:
        gamma: focusing parameter — higher = more focus on hard examples.
               gamma=2 is the original paper default; gamma=3 works better
               when class imbalance is extreme.
        class_weights: (C,) float tensor of per-class loss multipliers,
                       typically inverse-frequency normalised to mean=1.
                       When None falls back to uniform weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        # kept for backward compat — ignored when class_weights is provided
        alpha: float = 0.25,
        num_classes: int = 9,
    ):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # cross-entropy with optional per-class weighting
        ce_loss = F.cross_entropy(
            pred, target,
            weight=self.class_weights,
            reduction="none",
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al., CVPR 2019).

    Re-weights by the "effective number of samples" for each class:
        w_y = (1 - beta) / (1 - beta^n_y)
    where n_y is the sample count for class y and beta in [0, 1).

    This is a strictly better proxy for the occupied volume in feature space
    than raw inverse-frequency, and handles the 1000:1 imbalance in WM-811K
    (None vs Near-Full) far more gracefully.

    Label smoothing is built in because it provides consistent additional
    regularisation without requiring a separate loss composition.

    Args:
        samples_per_class: list of sample counts, one per class
        beta: volume proxy parameter — 0.9999 recommended for heavy imbalance
        gamma: focal modulating factor — can be reduced to 2.0 when CB handles
               the imbalance correction (vs 3.0 for plain focal)
        smoothing: label smoothing epsilon (0.1 recommended)
        num_classes: number of output classes
    """

    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        num_classes: int = 9,
    ):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        # normalise so mean weight == 1 (preserves gradient magnitude scale)
        weights = weights / weights.sum() * num_classes
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            pred, target,
            weight=self.class_weights,
            label_smoothing=self.smoothing,
            reduction="none",
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
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
