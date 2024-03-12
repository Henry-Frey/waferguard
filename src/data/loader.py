"""DataLoader factory with stratified splitting and class balancing."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.transforms import build_classifier_transforms
from src.data.wm811k import WM811KDataset, preprocess_wm811k
from src.utils.logging import get_logger

log = get_logger(__name__)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalised so their mean == 1.

    Useful as per-class alpha in FocalLoss.  A class with half as many
    samples as the mean gets weight 2.0; twice as many → weight 0.5.
    """
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )
    return weights


def _build_sampler(train_labels: np.ndarray, strategy: str) -> WeightedRandomSampler | None:
    """Build a WeightedRandomSampler for the given balance strategy.

    Strategies
    ----------
    oversample         Inverse-frequency; extreme (~123:1) ratio.
    sqrt_oversample    Sqrt-damped inverse-freq; ~11:1 ratio.
    focal_loss         No sampler — rely on FocalLoss class weighting.
    """
    counts = Counter(train_labels.tolist())

    if strategy == "oversample":
        sample_weights = [1.0 / counts[l] for l in train_labels]
    elif strategy == "sqrt_oversample":
        sample_weights = [1.0 / (counts[l] ** 0.5) for l in train_labels]
    else:
        return None

    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def build_classifier_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for WM-811K classification.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    wafer_maps, labels = preprocess_wm811k(
        cfg.wm811k.path,
        img_size=cfg.data.img_size,
        noise_filter=cfg.wm811k.noise_filter.enabled,
        window_size=cfg.wm811k.noise_filter.window_size,
        min_neighbors=cfg.wm811k.noise_filter.min_neighbors,
    )

    # Stratified train / val / test split
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg.wm811k.test_ratio, random_state=cfg.seed
    )
    train_val_idx, test_idx = next(sss1.split(wafer_maps, labels))

    val_ratio_adj = cfg.wm811k.val_ratio / (1 - cfg.wm811k.test_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adj, random_state=cfg.seed
    )
    train_idx, val_idx = next(
        sss2.split(wafer_maps[train_val_idx], labels[train_val_idx])
    )
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    log.info("data_split", train=len(train_idx), val=len(val_idx), test=len(test_idx))

    train_labels = labels[train_idx]

    train_tfms = build_classifier_transforms(cfg.data.img_size, is_train=True)
    val_tfms = build_classifier_transforms(cfg.data.img_size, is_train=False)

    train_ds = WM811KDataset(wafer_maps[train_idx], train_labels, cfg.data.img_size, train_tfms)
    val_ds = WM811KDataset(wafer_maps[val_idx], labels[val_idx], cfg.data.img_size, val_tfms)
    test_ds = WM811KDataset(wafer_maps[test_idx], labels[test_idx], cfg.data.img_size, val_tfms)

    strategy = cfg.wm811k.balance_strategy
    sampler = _build_sampler(train_labels, strategy)

    num_workers = cfg.data.num_workers
    pin_memory = cfg.data.pin_memory and torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.classifier.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.classifier.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.classifier.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader
