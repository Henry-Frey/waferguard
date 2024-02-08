"""Learning rate schedulers with warmup support."""

from __future__ import annotations

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        import math
        cosine_scale = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]
