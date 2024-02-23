"""Training loop for wafer defect pattern classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from omegaconf import DictConfig

from src.data.loader import build_classifier_loaders, compute_samples_per_class
from src.models.classifier import build_classifier
from src.training.losses import ClassBalancedFocalLoss, FocalLoss, LabelSmoothingLoss
from src.training.sam import SAM
from src.evaluation.metrics import ClassificationMetrics
from src.utils.config import parse_args_to_config
from src.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CutMix batch augmentation
# ---------------------------------------------------------------------------

def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix batch augmentation (Yun et al., 2019).

    Mixes rectangular patches between pairs of training images and
    interpolates labels proportionally to the replaced area.

    Returns:
        (mixed_images, labels_a, labels_b, lam)
        Loss = lam * criterion(logits, labels_a) + (1-lam) * criterion(logits, labels_b)
    """
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    perm = torch.randperm(B, device=images.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    return mixed, labels, labels[perm], lam


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ClassifierTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Resolve device from config; fall back to cuda:0 / cpu
        train_cfg = cfg.get("training", {})
        device_str = train_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        torch.cuda.set_device(self.device)   # pin default CUDA device for this process
        log.info("device", device=str(self.device),
                 gpu_name=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
                 vram_gb=f"{torch.cuda.get_device_properties(self.device).total_memory/1024**3:.1f}"
                 if self.device.type == "cuda" else "n/a")

        self.train_loader, self.val_loader, self.test_loader, class_weights = build_classifier_loaders(cfg)
        self.class_weights = class_weights.to(self.device)

        self.model = build_classifier(cfg).to(self.device)

        # Optional DataParallel across multiple GPUs (disabled in sota.yaml)
        if train_cfg.get("data_parallel", False) and torch.cuda.device_count() > 1:
            gpu_ids = list(train_cfg.get("gpu_ids", list(range(torch.cuda.device_count()))))
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
            log.info("data_parallel_enabled", gpus=gpu_ids)
        else:
            log.info("single_gpu_mode", device=str(self.device))

        # Need train labels for CB-Focal loss construction
        self.train_labels = self.train_loader.dataset.labels
        self.criterion = self._build_loss().to(self.device)
        self.optimizer = self._build_optimizer()

        sched_opt = (
            self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            sched_opt,
            T_max=cfg.classifier.epochs,
            eta_min=cfg.classifier.scheduler.min_lr,
        )
        self.scaler = GradScaler("cuda")
        self.metrics = ClassificationMetrics(cfg.wm811k.num_classes, cfg.wm811k.class_names)

        self.best_metric = 0.0
        self.patience_counter = 0
        self.artifact_dir = Path("artifacts")
        self.artifact_dir.mkdir(exist_ok=True)

        # CutMix config (defaults: off)
        cutmix_cfg = cfg.get("cutmix", {})
        self.cutmix_enabled = cutmix_cfg.get("enabled", False)
        self.cutmix_alpha = cutmix_cfg.get("alpha", 1.0)
        self.cutmix_prob = cutmix_cfg.get("prob", 0.5)

        # SAM interval: apply perturbation every N steps (normal AdamW otherwise)
        # Reduces overhead from 2× to ~10% while retaining most of the benefit.
        self.sam_interval = cfg.classifier.optimizer.get("sam_interval", 1)

    def _build_loss(self):
        name = self.cfg.classifier.loss.name
        strategy = self.cfg.wm811k.balance_strategy
        uses_sampler = strategy in ("oversample", "sqrt_oversample", "class_specific_oversample")
        weights = None if uses_sampler else self.class_weights

        if name == "cb_focal":
            samples_per_class = compute_samples_per_class(
                self.train_labels, self.cfg.wm811k.num_classes
            )
            return ClassBalancedFocalLoss(
                samples_per_class=samples_per_class,
                beta=self.cfg.classifier.loss.get("beta", 0.9999),
                gamma=self.cfg.classifier.loss.get("gamma", 2.0),
                smoothing=self.cfg.classifier.loss.get("smoothing", 0.1),
                num_classes=self.cfg.wm811k.num_classes,
            )
        if name == "focal":
            return FocalLoss(
                gamma=self.cfg.classifier.loss.gamma,
                class_weights=weights,
                num_classes=self.cfg.wm811k.num_classes,
            )
        if name == "label_smoothing":
            return LabelSmoothingLoss(num_classes=self.cfg.wm811k.num_classes)
        return torch.nn.CrossEntropyLoss(weight=weights)

    def _build_optimizer(self):
        opt_name = self.cfg.classifier.optimizer.get("name", "adamw")
        lr = self.cfg.classifier.optimizer.lr
        wd = self.cfg.classifier.optimizer.weight_decay
        base_opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        if opt_name == "sam":
            rho = self.cfg.classifier.optimizer.get("sam_rho", 0.05)
            adaptive = self.cfg.classifier.optimizer.get("sam_adaptive", False)
            log.info("optimizer_sam", rho=rho, adaptive=adaptive)
            return SAM(self.model.parameters(), base_optimizer=base_opt, rho=rho, adaptive=adaptive)

        return base_opt

    def _sam_update(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """Full two-step SAM update with AMP support.

        unscale_() can only be called once per optimizer per scaler.update()
        cycle.  We skip it before first_step — the perturbation direction is
        preserved with scaled gradients because grad and grad_norm both scale
        by the same GradScaler factor, so it cancels in e_w = rho*grad/norm.
        unscale_() is called only before second_step so the base optimizer
        receives proper float32 gradients.
        """
        # --- pass 1: compute perturbation direction (scaled grads OK) ---
        with autocast("cuda"):
            loss = self.criterion(self.model(images), labels)
        self.scaler.scale(loss).backward()
        self.optimizer.first_step(zero_grad=True)

        # --- pass 2: descent step at perturbed weights ---
        with autocast("cuda"):
            loss2 = self.criterion(self.model(images), labels)
        self.scaler.scale(loss2).backward()
        self.scaler.unscale_(self.optimizer)
        self.optimizer.second_step(zero_grad=True)
        self.scaler.update()
        return loss.item()

    def _normal_update(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """Standard single-pass update with AMP."""
        base_opt = (
            self.optimizer.base_optimizer if isinstance(self.optimizer, SAM) else self.optimizer
        )
        with autocast("cuda"):
            loss = self.criterion(self.model(images), labels)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(base_opt)
        self.scaler.update()
        return loss.item()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        use_sam = isinstance(self.optimizer, SAM)

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Optional CutMix batch augmentation
            if self.cutmix_enabled and np.random.random() < self.cutmix_prob:
                images, labels_a, labels_b, lam = cutmix_batch(images, labels, self.cutmix_alpha)
                crit = self.criterion
                # Temporarily bind mixed-label loss for SAM/normal update paths
                self.criterion = lambda logits, _: (
                    lam * crit(logits, labels_a) + (1 - lam) * crit(logits, labels_b)
                )
                do_sam = use_sam and step % self.sam_interval == 0
                loss_val = self._sam_update(images, labels) if do_sam else self._normal_update(images, labels)
                self.criterion = crit
            else:
                do_sam = use_sam and step % self.sam_interval == 0
                loss_val = self._sam_update(images, labels) if do_sam else self._normal_update(images, labels)

            total_loss += loss_val

        self.scheduler.step()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader, split: str = "val") -> dict[str, float]:
        self.model.eval()
        self.metrics.reset()

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            logits = self.model(images)
            self.metrics.update(logits, labels)

        results = self.metrics.compute()
        log.info(f"{split}_metrics", **{k: f"{v:.4f}" for k, v in results.items()})
        return results

    def save_checkpoint(self, epoch: int, metric: float) -> None:
        opt_state = (
            self.optimizer.base_optimizer.state_dict()
            if isinstance(self.optimizer, SAM)
            else self.optimizer.state_dict()
        )
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": opt_state,
            "metric": metric,
        }
        torch.save(state, self.artifact_dir / "classifier_last.pt")
        if metric > self.best_metric:
            self.best_metric = metric
            torch.save(state, self.artifact_dir / "classifier_best.pt")
            log.info("new_best_model", f1_macro=f"{metric:.4f}")

    def fit(self) -> None:
        log.info("classifier_training_start", epochs=self.cfg.classifier.epochs)

        for epoch in range(1, self.cfg.classifier.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "val")

            metric = val_metrics["f1_macro"]
            self.save_checkpoint(epoch, metric)

            log.info("epoch_complete", epoch=epoch, train_loss=f"{train_loss:.4f}",
                     f1_macro=f"{metric:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")

            if metric <= self.best_metric:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.classifier.early_stopping.patience:
                    log.info("early_stopping", epoch=epoch)
                    break
            else:
                self.patience_counter = 0

        # final test evaluation
        test_metrics = self.evaluate(self.test_loader, "test")
        log.info("training_complete", best_f1=f"{self.best_metric:.4f}", test_metrics=test_metrics)


if __name__ == "__main__":
    setup_logging()
    cfg = parse_args_to_config()
    trainer = ClassifierTrainer(cfg)
    trainer.fit()
