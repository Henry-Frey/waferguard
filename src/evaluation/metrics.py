"""Metrics for classification and anomaly detection."""

from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)


class ClassificationMetrics:
    """Accumulates predictions and computes classification metrics."""

    def __init__(self, num_classes: int, class_names: list[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def reset(self) -> None:
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=1).cpu()
        preds = logits.argmax(dim=1).cpu()
        self.all_preds.append(preds)
        self.all_labels.append(labels.cpu())
        self.all_probs.append(probs)

    def compute(self) -> dict[str, float]:
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()

        results = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        }

        # per-class F1
        f1_per = f1_score(labels, preds, average=None, zero_division=0)
        for i, name in enumerate(self.class_names):
            if i < len(f1_per):
                results[f"f1_{name}"] = f1_per[i]

        return results

    def confusion_matrix(self) -> np.ndarray:
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()
        return confusion_matrix(labels, preds)


class AnomalyMetrics:
    """Metrics for anomaly detection: image-level AUROC and pixel-level AUROC."""

    @staticmethod
    def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
        return roc_auc_score(labels, scores)

    @staticmethod
    def pixel_auroc(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> float:
        return roc_auc_score(gt_masks.flatten(), anomaly_maps.flatten())

    @staticmethod
    def optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
        """Find threshold maximizing Youden's J statistic."""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_stat = tpr - fpr
        return thresholds[np.argmax(j_stat)]
