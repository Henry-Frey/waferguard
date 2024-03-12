"""Bayesian hyperparameter optimisation for WaferClassifier.

Uses Optuna with the TPE (Tree-structured Parzen Estimator) sampler —
a form of Bayesian optimisation that models the objective with kernel
density estimators and samples from promising regions.

Key hyperparameters tuned
--------------------------
lr             Learning rate           [1e-5, 1e-2] log-scale
weight_decay   L2 regulariser          [1e-6, 1e-2] log-scale
dropout        Head dropout rate       [0.10, 0.50]
focal_alpha    FocalLoss scalar alpha  [0.10, 0.50]
focal_gamma    FocalLoss focusing exp  [1.0,  5.0]
batch_size     Mini-batch size         {64, 128, 256, 512}
warmup_epochs  LR warmup length        [1, 8]

Each trial trains for `hpo.trial_epochs` (default 15) instead of the
full 50 to keep wall-clock time reasonable.  Median pruning kills trials
that fall below the running median after each epoch.

Top-K trial checkpoints are saved to `hpo.artifact_dir` for ensemble use.

Usage
-----
    python -m src.training.hpo                          # uses base.yaml + hpo.yaml
    python -m src.training.hpo hpo.n_trials=30          # override from CLI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.loader import build_classifier_loaders
from src.evaluation.metrics import ClassificationMetrics
from src.models.classifier import WaferClassifier
from src.training.losses import FocalLoss, LabelSmoothingLoss
from src.training.scheduler import WarmupCosineScheduler
from src.utils.logging import get_logger

log = get_logger(__name__)

# ── Search-space boundaries ───────────────────────────────────────────────────
_LR_BOUNDS = (1e-5, 1e-2)
_WD_BOUNDS = (1e-6, 1e-2)
_DROPOUT_BOUNDS = (0.10, 0.50)
_ALPHA_BOUNDS = (0.10, 0.50)
_GAMMA_BOUNDS = (1.0, 5.0)
_BATCH_SIZES = [64, 128, 256, 512]
_WARMUP_BOUNDS = (1, 8)


def _suggest(trial: optuna.Trial) -> dict[str, Any]:
    """Draw a hyperparameter configuration from the Optuna trial."""
    return {
        "lr": trial.suggest_float("lr", *_LR_BOUNDS, log=True),
        "weight_decay": trial.suggest_float("weight_decay", *_WD_BOUNDS, log=True),
        "dropout": trial.suggest_float("dropout", *_DROPOUT_BOUNDS),
        "focal_alpha": trial.suggest_float("focal_alpha", *_ALPHA_BOUNDS),
        "focal_gamma": trial.suggest_float("focal_gamma", *_GAMMA_BOUNDS),
        "batch_size": trial.suggest_categorical("batch_size", _BATCH_SIZES),
        "warmup_epochs": trial.suggest_int("warmup_epochs", *_WARMUP_BOUNDS),
    }


def _patch_cfg(base_cfg: DictConfig, params: dict[str, Any], trial_epochs: int) -> DictConfig:
    """Return a modified copy of base_cfg with the trial's hyperparameters."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.classifier.optimizer.lr = params["lr"]
    cfg.classifier.optimizer.weight_decay = params["weight_decay"]
    cfg.classifier.loss.alpha = params["focal_alpha"]
    cfg.classifier.loss.gamma = params["focal_gamma"]
    cfg.classifier.batch_size = params["batch_size"]
    cfg.classifier.scheduler.warmup_epochs = params["warmup_epochs"]
    cfg.classifier.epochs = trial_epochs
    cfg.classifier.early_stopping.patience = max(3, trial_epochs // 4)
    return cfg


def _run_trial(
    trial: optuna.Trial,
    base_cfg: DictConfig,
    device: torch.device,
    artifact_dir: Path,
    trial_epochs: int,
) -> float:
    """Train one Optuna trial and return validation F1-macro.

    Checkpoints the best epoch state so ensemble can reload it later.
    Raises ``optuna.exceptions.TrialPruned`` when the median pruner fires.
    """
    params = _suggest(trial)
    cfg = _patch_cfg(base_cfg, params, trial_epochs)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_classifier_loaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WaferClassifier(
        backbone_name=cfg.classifier.backbone,
        num_classes=cfg.wm811k.num_classes,
        pretrained=cfg.classifier.pretrained,
        dropout=params["dropout"],
    ).to(device)

    import sys
    if sys.platform != "win32" and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass  # compile is best-effort; keep going without it

    # ── Optimisation ──────────────────────────────────────────────────────────
    criterion: torch.nn.Module
    if cfg.classifier.loss.name == "focal":
        criterion = FocalLoss(
            alpha=params["focal_alpha"],
            gamma=params["focal_gamma"],
            num_classes=cfg.wm811k.num_classes,
        )
    elif cfg.classifier.loss.name == "label_smoothing":
        criterion = LabelSmoothingLoss(num_classes=cfg.wm811k.num_classes)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=params["warmup_epochs"],
        total_epochs=trial_epochs,
        min_lr=cfg.classifier.scheduler.min_lr,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics_tracker = ClassificationMetrics(cfg.wm811k.num_classes, cfg.wm811k.class_names)

    best_f1 = 0.0
    patience = cfg.classifier.early_stopping.patience
    no_improve = 0

    for epoch in range(trial_epochs):
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

        # Report to Optuna for pruning
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            # Save checkpoint for ensemble
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {
                    "trial_number": trial.number,
                    "params": params,
                    "epoch": epoch,
                    "f1_macro": best_f1,
                    "model_state": raw_model.state_dict(),
                    "backbone": cfg.classifier.backbone,
                },
                artifact_dir / f"hpo_trial_{trial.number:04d}.pt",
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    log.info(
        "trial_done",
        trial=trial.number,
        best_f1=f"{best_f1:.4f}",
        params={k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in params.items()},
    )
    return best_f1


def run_hpo(cfg: DictConfig) -> dict[str, Any]:
    """Run the full Bayesian HPO study.

    Returns a dict with:
        best_params       — hyperparameters of the best trial
        best_value        — val F1-macro of the best trial
        top_trial_numbers — trial indices to use for ensemble (top-K)
        study             — the raw optuna.Study object
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_trials: int = cfg.hpo.n_trials
    trial_epochs: int = cfg.hpo.trial_epochs
    artifact_dir = Path(cfg.hpo.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    log.info("hpo_start", device=str(device), n_trials=n_trials, trial_epochs=trial_epochs)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.hpo.get("n_startup_trials", 5),
        n_warmup_steps=cfg.hpo.get("n_warmup_steps", 3),
    )

    study = optuna.create_study(
        study_name=cfg.hpo.get("study_name", "waferguard_hpo"),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: _run_trial(trial, cfg, device, artifact_dir, trial_epochs),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    log.info("hpo_complete", best_trial=best.number, best_f1=f"{best.value:.4f}")
    log.info("best_params", **best.params)

    # Save best hyperparameters
    best_params_path = artifact_dir / "best_hparams.yaml"
    OmegaConf.save(OmegaConf.create(best.params), best_params_path)
    log.info("best_hparams_saved", path=str(best_params_path))

    # Rank completed trials for ensemble selection
    top_k: int = cfg.hpo.get("ensemble_top_k", 3)
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(finished, key=lambda t: t.value, reverse=True)[:top_k]
    top_trial_numbers = [t.number for t in top_trials]
    log.info("ensemble_top_k", k=top_k, trial_numbers=top_trial_numbers)

    return {
        "best_params": best.params,
        "best_value": best.value,
        "top_trial_numbers": top_trial_numbers,
        "study": study,
    }


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    from src.utils.config import load_config
    from src.utils.logging import setup_logging

    setup_logging()
    # Merge base config with HPO-specific config, then apply any CLI overrides.
    _base = load_config("configs/base.yaml")
    _hpo_cfg = load_config("configs/hpo.yaml")
    _cfg = OmegaConf.merge(_base, _hpo_cfg)

    # Apply any remaining dotlist overrides from CLI (e.g. hpo.n_trials=30)
    _overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    if _overrides:
        _cfg = OmegaConf.merge(_cfg, OmegaConf.from_dotlist(_overrides))

    run_hpo(_cfg)
