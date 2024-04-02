"""Resume top-K HPO survivors to full training for ensemble use.

After the HPO search, this script:
  1. Loads the top-K trial checkpoints from artifacts/hpo/ (ranked by val F1).
  2. Resumes each from its saved epoch, running to completion (cfg.classifier.epochs
     epochs total) with early stopping still active.
  3. Saves the fully-trained members to artifacts/ensemble/member_XX.pt.

The ensemble is then built from these completed models rather than the
partial trial checkpoints, giving it properly trained members.

Usage
-----
    python -m src.training.train_ensemble_members          # default top-3
    python -m src.training.train_ensemble_members hpo.ensemble_top_k=5
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.loader import build_classifier_loaders
from src.evaluation.metrics import ClassificationMetrics
from src.models.classifier import WaferClassifier
from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.training.scheduler import WarmupCosineScheduler
from src.utils.logging import get_logger

log = get_logger(__name__)


def _load_top_k(hpo_dir: Path, top_k: int) -> list[dict]:
    """Load and rank HPO trial checkpoints by saved val F1-macro."""
    ckpt_files = sorted(hpo_dir.glob("hpo_trial_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No HPO trial checkpoints found in {hpo_dir}")

    ranked = sorted(
        ckpt_files,
        key=lambda p: torch.load(p, map_location="cpu", weights_only=False).get("f1_macro", 0.0),
        reverse=True,
    )
    selected = ranked[:top_k]

    checkpoints = []
    for path in selected:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        log.info(
            "selected_trial",
            file=path.name,
            trial=ckpt.get("trial_number", "?"),
            hpo_f1=f"{ckpt.get('f1_macro', 0.0):.4f}",
            hpo_epoch=ckpt.get("epoch", "?"),
        )
        checkpoints.append(ckpt)
    return checkpoints


def _resume_to_completion(
    ckpt: dict,
    cfg: DictConfig,
    member_idx: int,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Continue training a single HPO checkpoint to cfg.classifier.epochs total.

    The scheduler is re-initialised at the saved epoch position so the LR
    continues smoothly on the cosine curve rather than restarting.
    The optimizer is recreated fresh (momentum buffers lost, acceptable for
    resumption — the scheduler corrects the LR within a couple of steps).
    """
    params: dict = ckpt["params"]
    backbone: str = ckpt.get("backbone", cfg.classifier.backbone)
    dropout: float = params.get("dropout", cfg.classifier.get("dropout", 0.3))
    hpo_epoch: int = ckpt["epoch"]          # last completed epoch in HPO (0-indexed)
    total_epochs: int = cfg.classifier.epochs
    start_epoch: int = hpo_epoch + 1        # first epoch still to train

    output_path = output_dir / f"member_{member_idx:02d}.pt"

    if start_epoch >= total_epochs:
        # Trial already ran to the full budget — no extra training needed.
        log.info("already_complete", member=member_idx, epoch=hpo_epoch)
        torch.save(
            {
                "member_idx": member_idx,
                "params": params,
                "backbone": backbone,
                "dropout": dropout,
                "f1_macro": ckpt.get("f1_macro", 0.0),
                "model_state": ckpt["model_state"],
            },
            output_path,
        )
        return

    log.info(
        "resuming",
        member=member_idx,
        backbone=backbone,
        hpo_epoch=hpo_epoch,
        remaining_epochs=total_epochs - start_epoch,
        params={k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in params.items()},
    )

    # ── Patch cfg with this trial's hyperparameters ───────────────────────────
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.classifier.optimizer.lr = params["lr"]
    cfg.classifier.optimizer.weight_decay = params["weight_decay"]
    cfg.classifier.loss.alpha = params["focal_alpha"]
    cfg.classifier.loss.gamma = params["focal_gamma"]
    cfg.classifier.batch_size = params["batch_size"]
    cfg.classifier.scheduler.warmup_epochs = params["warmup_epochs"]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_classifier_loaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WaferClassifier(
        backbone_name=backbone,
        num_classes=cfg.wm811k.num_classes,
        pretrained=False,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ── Loss ──────────────────────────────────────────────────────────────────
    if cfg.classifier.loss.name == "focal":
        criterion: torch.nn.Module = FocalLoss(
            alpha=params["focal_alpha"],
            gamma=params["focal_gamma"],
            num_classes=cfg.wm811k.num_classes,
        )
    elif cfg.classifier.loss.name == "label_smoothing":
        criterion = LabelSmoothingLoss(num_classes=cfg.wm811k.num_classes)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    # ── Scheduler — resume at the correct cosine position ────────────────────
    # _LRScheduler.__init__ calls step() once, advancing last_epoch by 1.
    # Passing last_epoch = hpo_epoch means after init last_epoch = start_epoch,
    # so the first training step uses the LR for start_epoch on the cosine curve.
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=params["warmup_epochs"],
        total_epochs=total_epochs,
        min_lr=cfg.classifier.scheduler.min_lr,
        last_epoch=hpo_epoch,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics_tracker = ClassificationMetrics(cfg.wm811k.num_classes, cfg.wm811k.class_names)

    best_f1: float = ckpt.get("f1_macro", 0.0)
    best_state: dict = {k: v.cpu().clone() for k, v in ckpt["model_state"].items()}
    patience: int = cfg.classifier.early_stopping.patience
    no_improve: int = 0

    for epoch in range(start_epoch, total_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        metrics_tracker.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                metrics_tracker.update(logits, labels)

        val_f1 = metrics_tracker.compute()["f1_macro"]
        log.info(
            "epoch",
            member=member_idx,
            epoch=epoch,
            f1_macro=f"{val_f1:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            log.info("new_best", member=member_idx, f1_macro=f"{best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("early_stopping", member=member_idx, epoch=epoch)
                break

    torch.save(
        {
            "member_idx": member_idx,
            "params": params,
            "backbone": backbone,
            "dropout": dropout,
            "f1_macro": best_f1,
            "model_state": best_state,
        },
        output_path,
    )
    log.info("member_saved", path=str(output_path), final_f1=f"{best_f1:.4f}")


def train_ensemble_members(cfg: DictConfig) -> None:
    """Load top-K HPO checkpoints and train each to completion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hpo_dir = Path(cfg.hpo.artifact_dir)
    output_dir = Path(cfg.hpo.get("ensemble_dir", "artifacts/ensemble"))
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k: int = cfg.hpo.get("ensemble_top_k", 3)

    log.info(
        "ensemble_training_start",
        device=str(device),
        top_k=top_k,
        hpo_dir=str(hpo_dir),
        output_dir=str(output_dir),
    )

    top_ckpts = _load_top_k(hpo_dir, top_k)
    for i, ckpt in enumerate(top_ckpts):
        _resume_to_completion(ckpt, cfg, member_idx=i, output_dir=output_dir, device=device)

    log.info("ensemble_training_complete", n_members=len(top_ckpts), output_dir=str(output_dir))


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    from src.utils.config import load_config
    from src.utils.logging import setup_logging

    setup_logging()
    _base = load_config("configs/base.yaml")
    _hpo_cfg = load_config("configs/hpo.yaml")
    _cfg = OmegaConf.merge(_base, _hpo_cfg)

    _overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    if _overrides:
        _cfg = OmegaConf.merge(_cfg, OmegaConf.from_dotlist(_overrides))

    train_ensemble_members(_cfg)
