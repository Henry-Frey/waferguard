"""DataLoader factory with stratified splitting and class balancing."""

from __future__ import annotations

import numpy as np
import torch
from collections import Counter

from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.transforms import build_classifier_transforms
from src.data.wm811k import WM811KDataset, WM811KDatasetV2, preprocess_wm811k
from src.utils.logging import get_logger

log = get_logger(__name__)

# Class indices for targeted oversampling (roadmap: 3-5× for Scratch and Loc)
_SCRATCH_CLASS = 7
_LOC_CLASS = 4
_TARGETED_OVERSAMPLE_CLASSES = {_SCRATCH_CLASS, _LOC_CLASS}
_TARGETED_OVERSAMPLE_BOOST = 4.0   # 4× extra weight for Scratch and Loc


def compute_class_weights(
    labels: np.ndarray, num_classes: int
) -> torch.Tensor:
    """Inverse-frequency class weights normalised so their mean == 1.

    Used as per-class alpha in FocalLoss to replace the naive scalar alpha.
    A class with half as many samples as the average gets weight 2.0;
    a class with twice as many gets weight 0.5.
    """
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )
    return weights


def compute_samples_per_class(labels: np.ndarray, num_classes: int) -> list[int]:
    """Return sample count per class (required by ClassBalancedFocalLoss)."""
    counts = Counter(labels.tolist())
    return [counts.get(i, 1) for i in range(num_classes)]


def _build_sampler(
    train_labels: np.ndarray,
    strategy: str,
) -> WeightedRandomSampler | None:
    """Build a WeightedRandomSampler according to the balance strategy.

    Strategies:
        oversample              — pure inverse-frequency; extreme (~123:1) ratio
        sqrt_oversample         — sqrt-damped inverse-freq; ~11:1 ratio
        class_specific_oversample — sqrt_oversample with 4× boost for Scratch
                                    and Loc (the two lowest-F1 classes)
    """
    counts = Counter(train_labels.tolist())

    if strategy == "oversample":
        sample_weights = [1.0 / counts[l] for l in train_labels]
    elif strategy == "sqrt_oversample":
        sample_weights = [1.0 / (counts[l] ** 0.5) for l in train_labels]
    elif strategy == "class_specific_oversample":
        # sqrt base weight + targeted boost for Scratch and Loc
        sample_weights = []
        for l in train_labels:
            w = 1.0 / (counts[l] ** 0.5)
            if l in _TARGETED_OVERSAMPLE_CLASSES:
                w *= _TARGETED_OVERSAMPLE_BOOST
            sample_weights.append(w)
    else:
        return None

    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def _build_dataset(
    wafer_maps: np.ndarray,
    labels: np.ndarray,
    img_size: int,
    transforms,
    cfg: DictConfig,
) -> WM811KDataset | WM811KDatasetV2:
    """Construct the appropriate dataset class from config."""
    features_cfg = cfg.get("features", {})
    use_radon = features_cfg.get("use_radon", False)
    use_distance = features_cfg.get("use_distance_from_center", False)

    if use_radon or use_distance:
        return WM811KDatasetV2(
            wafer_maps=wafer_maps,
            labels=labels,
            img_size=img_size,
            transforms=transforms,
            use_radon=use_radon,
            use_distance=use_distance,
            radon_angles=features_cfg.get("radon_angles", 36),
        )
    return WM811KDataset(wafer_maps, labels, img_size, transforms)


def build_classifier_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Build train/val/test loaders for WM-811K classification.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
        class_weights is a (C,) tensor of inverse-frequency weights for FocalLoss.
    """
    wafer_maps, labels = preprocess_wm811k(
        cfg.wm811k.path,
        img_size=cfg.data.img_size,
        noise_filter=cfg.wm811k.noise_filter.enabled,
        window_size=cfg.wm811k.noise_filter.window_size,
        min_neighbors=cfg.wm811k.noise_filter.min_neighbors,
    )

    # stratified train/val/test split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=cfg.wm811k.test_ratio, random_state=cfg.seed)
    train_val_idx, test_idx = next(sss1.split(wafer_maps, labels))

    val_ratio_adj = cfg.wm811k.val_ratio / (1 - cfg.wm811k.test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adj, random_state=cfg.seed)
    train_idx, val_idx = next(sss2.split(wafer_maps[train_val_idx], labels[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    log.info("data_split", train=len(train_idx), val=len(val_idx), test=len(test_idx))

    train_labels = labels[train_idx]
    class_weights = compute_class_weights(train_labels, cfg.wm811k.num_classes)
    log.info(
        "class_weights",
        **{cfg.wm811k.class_names[i]: f"{class_weights[i]:.3f}" for i in range(cfg.wm811k.num_classes)},
    )

    extra_ch = sum([
        cfg.get("features", {}).get("use_radon", False),
        cfg.get("features", {}).get("use_distance_from_center", False),
    ])
    train_tfms = build_classifier_transforms(cfg.data.img_size, is_train=True, extra_channels=extra_ch)
    val_tfms = build_classifier_transforms(cfg.data.img_size, is_train=False, extra_channels=extra_ch)

    train_ds = _build_dataset(wafer_maps[train_idx], train_labels, cfg.data.img_size, train_tfms, cfg)
    val_ds = _build_dataset(wafer_maps[val_idx], labels[val_idx], cfg.data.img_size, val_tfms, cfg)
    test_ds = _build_dataset(wafer_maps[test_idx], labels[test_idx], cfg.data.img_size, val_tfms, cfg)

    strategy = cfg.wm811k.balance_strategy
    sampler = _build_sampler(train_labels, strategy)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.classifier.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.classifier.batch_size * 2,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.classifier.batch_size * 2,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader, test_loader, class_weights
