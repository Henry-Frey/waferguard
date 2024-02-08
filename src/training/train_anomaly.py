"""Build PatchCore memory bank from defect-free wafer data."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.data.transforms import build_anomaly_transforms
from src.data.wm811k import WM811KDataset, preprocess_wm811k, LABEL_MAP
from src.models.patchcore import PatchCore
from src.utils.config import parse_args_to_config
from src.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def build_normal_loader(cfg: DictConfig) -> DataLoader:
    """Build a DataLoader of only defect-free wafer maps."""
    wafer_maps, labels = preprocess_wm811k(
        cfg.wm811k.path, img_size=cfg.anomaly.img_size,
        noise_filter=cfg.wm811k.noise_filter.enabled,
    )
    # filter to "None" class only (defect-free)
    none_label = LABEL_MAP["none"]
    mask = labels == none_label
    log.info("normal_samples", count=mask.sum())

    tfms = build_anomaly_transforms(cfg.anomaly.img_size, is_train=True)
    ds = WM811KDataset(wafer_maps[mask], labels[mask], cfg.anomaly.img_size, tfms)

    return DataLoader(
        ds, batch_size=cfg.anomaly.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )


def main():
    setup_logging()
    cfg = parse_args_to_config()

    loader = build_normal_loader(cfg)

    patchcore = PatchCore(
        backbone_name=cfg.anomaly.backbone,
        layers=cfg.anomaly.layers,
        coreset_ratio=cfg.anomaly.coreset_ratio,
        k_nearest=cfg.anomaly.k_nearest,
    )

    patchcore.fit(loader)

    out_path = Path("artifacts") / "memory_bank.pt"
    out_path.parent.mkdir(exist_ok=True)
    patchcore.save(str(out_path))
    log.info("memory_bank_saved", path=str(out_path))


if __name__ == "__main__":
    main()
