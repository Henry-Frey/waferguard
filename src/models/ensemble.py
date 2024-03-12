"""Soft-voting ensemble of WaferClassifier models.

Loads the top-K checkpoint files produced by the HPO study, runs each
model independently on a batch, then averages their softmax probabilities
(soft voting) to produce the final prediction.

Soft voting outperforms hard (majority) voting when model confidences
are calibrated, which they generally are after FocalLoss training.

Usage
-----
    # After running HPO:
    python -m src.models.ensemble                       # uses hpo.yaml defaults
    python -m src.models.ensemble hpo.ensemble_top_k=5  # use top-5 models
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.classifier import WaferClassifier
from src.utils.logging import get_logger

log = get_logger(__name__)


class WaferEnsemble:
    """Soft-voting ensemble of WaferClassifier checkpoints.

    Args:
        checkpoint_paths: Paths to .pt checkpoint files (order doesn't matter).
        cfg: Base config — used for ``wm811k.num_classes`` and class names.
        device: Inference device.  Defaults to CUDA if available.
    """

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        cfg: DictConfig,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names: list[str] = list(cfg.wm811k.class_names)
        self.models: list[torch.nn.Module] = []

        for ckpt_path in checkpoint_paths:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            # Checkpoint stores backbone / dropout so each model may differ
            backbone = ckpt.get("backbone", cfg.classifier.backbone)
            params = ckpt.get("params", {})
            dropout = params.get("dropout", cfg.classifier.get("dropout", 0.3))

            model = WaferClassifier(
                backbone_name=backbone,
                num_classes=cfg.wm811k.num_classes,
                pretrained=False,
                dropout=dropout,
            )
            model.load_state_dict(ckpt["model_state"])
            model.to(self.device).eval()
            self.models.append(model)

            f1 = ckpt.get("f1_macro", ckpt.get("metric", float("nan")))
            log.info(
                "checkpoint_loaded",
                path=Path(ckpt_path).name,
                backbone=backbone,
                f1_macro=f"{f1:.4f}",
            )

        log.info("ensemble_ready", n_models=len(self.models))

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Averaged softmax probabilities over all member models.

        Args:
            images: (B, C, H, W) float tensor on any device.

        Returns:
            (B, num_classes) probability tensor (sums to 1 per sample).
        """
        images = images.to(self.device)
        all_probs = torch.stack(
            [F.softmax(model(images), dim=1) for model in self.models]
        )  # (K, B, C)
        return all_probs.mean(dim=0)  # (B, C)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted class indices and confidences.

        Returns:
            class_indices: (B,) int64 tensor
            confidences:   (B,) float tensor — max probability across classes
        """
        probs = self.predict_proba(images)
        confidences, class_indices = probs.max(dim=1)
        return class_indices, confidences

    def predict_named(self, images: torch.Tensor) -> list[dict]:
        """Human-readable predictions for a batch.

        Returns:
            List of dicts with keys ``class``, ``confidence``, ``probs``.
        """
        probs = self.predict_proba(images)
        confidences, class_indices = probs.max(dim=1)
        results = []
        for i in range(len(class_indices)):
            results.append(
                {
                    "class": self.class_names[class_indices[i].item()],
                    "confidence": confidences[i].item(),
                    "probs": {
                        name: probs[i, j].item()
                        for j, name in enumerate(self.class_names)
                    },
                }
            )
        return results

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def from_hpo_artifacts(
        cls,
        artifact_dir: str | Path,
        top_k: int,
        cfg: DictConfig,
        device: torch.device | None = None,
    ) -> "WaferEnsemble":
        """Load the top-K checkpoints ranked by saved val F1-macro.

        Args:
            artifact_dir: Directory containing ``hpo_trial_XXXX.pt`` files.
            top_k:        Number of models to include in the ensemble.
            cfg:          Base configuration.
            device:       Inference device.

        Returns:
            Initialised WaferEnsemble.
        """
        artifact_dir = Path(artifact_dir)
        ckpt_files = sorted(artifact_dir.glob("hpo_trial_*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No HPO checkpoints found in {artifact_dir}")

        # Rank by saved f1_macro (descending)
        ranked = sorted(
            ckpt_files,
            key=lambda p: torch.load(p, map_location="cpu", weights_only=False).get(
                "f1_macro", 0.0
            ),
            reverse=True,
        )
        selected = ranked[:top_k]
        log.info(
            "ensemble_from_hpo",
            total_available=len(ckpt_files),
            top_k=top_k,
            selected=[p.name for p in selected],
        )
        return cls(selected, cfg, device)

    @classmethod
    def from_best_checkpoint(
        cls, artifact_dir: str | Path, cfg: DictConfig, device: torch.device | None = None
    ) -> "WaferEnsemble":
        """Convenience: single-model 'ensemble' from classifier_best.pt."""
        path = Path(artifact_dir) / "classifier_best.pt"
        return cls([path], cfg, device)


# ── Entry point ───────────────────────────────────────────────────────────────

def _evaluate_ensemble(cfg: DictConfig) -> None:
    """Evaluate the HPO ensemble on the test split and print metrics."""
    from src.data.loader import build_classifier_loaders
    from src.evaluation.metrics import ClassificationMetrics
    from src.utils.logging import setup_logging

    setup_logging()

    artifact_dir = Path(cfg.hpo.artifact_dir)
    top_k: int = cfg.hpo.get("ensemble_top_k", 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble = WaferEnsemble.from_hpo_artifacts(artifact_dir, top_k, cfg, device)

    _, _, test_loader = build_classifier_loaders(cfg)
    metrics_tracker = ClassificationMetrics(cfg.wm811k.num_classes, cfg.wm811k.class_names)

    ensemble_metrics = ClassificationMetrics(cfg.wm811k.num_classes, cfg.wm811k.class_names)
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"]
            labels = batch["label"].to(device)
            probs = ensemble.predict_proba(images)
            # Pass raw logits-equivalent (log of probs) so metrics work the same
            ensemble_metrics.update(torch.log(probs + 1e-8), labels)

    results = ensemble_metrics.compute()
    log.info("ensemble_test_metrics", **{k: f"{v:.4f}" for k, v in results.items()})


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    from src.utils.config import load_config

    _base = load_config("configs/base.yaml")
    _hpo_cfg = load_config("configs/hpo.yaml")
    _cfg = OmegaConf.merge(_base, _hpo_cfg)

    _overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    if _overrides:
        _cfg = OmegaConf.merge(_cfg, OmegaConf.from_dotlist(_overrides))

    _evaluate_ensemble(_cfg)
