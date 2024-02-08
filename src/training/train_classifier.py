"""Training loop for wafer defect pattern classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig

from src.data.loader import build_classifier_loaders
from src.models.classifier import WaferClassifier
from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.evaluation.metrics import ClassificationMetrics
from src.utils.config import parse_args_to_config
from src.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


class ClassifierTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader, self.val_loader, self.test_loader = build_classifier_loaders(cfg)

        self.model = WaferClassifier(
            backbone_name=cfg.classifier.backbone,
            num_classes=cfg.wm811k.num_classes,
            pretrained=cfg.classifier.pretrained,
        ).to(self.device)

        self.criterion = self._build_loss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.classifier.optimizer.lr,
            weight_decay=cfg.classifier.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.classifier.epochs,
            eta_min=cfg.classifier.scheduler.min_lr,
        )
        self.scaler = GradScaler()
        self.metrics = ClassificationMetrics(cfg.wm811k.num_classes, cfg.wm811k.class_names)

        self.best_metric = 0.0
        self.patience_counter = 0
        self.artifact_dir = Path("artifacts")
        self.artifact_dir.mkdir(exist_ok=True)

    def _build_loss(self):
        name = self.cfg.classifier.loss.name
        if name == "focal":
            return FocalLoss(
                alpha=self.cfg.classifier.loss.alpha,
                gamma=self.cfg.classifier.loss.gamma,
                num_classes=self.cfg.wm811k.num_classes,
            )
        elif name == "label_smoothing":
            return LabelSmoothingLoss(num_classes=self.cfg.wm811k.num_classes)
        return torch.nn.CrossEntropyLoss()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

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
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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
