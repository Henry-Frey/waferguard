"""Evaluate classifier on the held-out test set and print a leaderboard.

Usage:
    python scripts/evaluate_testset.py
    python scripts/evaluate_testset.py classifier.batch_size=512
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

# Force UTF-8 stdout on Windows so box-drawing chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ensure src/ is on PYTHONPATH when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

from src.data.loader import build_classifier_loaders
from src.models.classifier import WaferClassifier
from src.utils.config import parse_args_to_config
from src.utils.logging import setup_logging, get_logger

log = get_logger(__name__)

# ── Literature baselines on WM-811K ──────────────────────────────────────────
# Accuracy / macro-F1 from published papers (approximate / reported values).
# Sources:
#   Wu et al. 2014  — original WM-811K paper (hand-crafted Radon features + SVM)
#   CNN-baseline    — standard single-branch CNN (no class balancing)
#   WaPIRL 2021     — self-supervised contrastive on WM-811K
#   MCNN-WaferNet   — multi-scale CNN reported in several 2022 papers
#   WaferVGG-16     — fine-tuned VGG16, reported ~97% accuracy
BASELINES = [
    {"Method":            "Wu et al. 2014 (Radon+SVM)",
     "Backbone":          "Hand-crafted",
     "Macro F1":          0.620,
     "Accuracy":          0.880,
     "Note":              "original paper"},
    {"Method":            "Plain CNN (no balancing)",
     "Backbone":          "Custom 3-conv",
     "Macro F1":          0.701,
     "Accuracy":          0.936,
     "Note":              "na\u00efve baseline"},
    {"Method":            "VGG-16 fine-tune",
     "Backbone":          "VGG-16",
     "Macro F1":          0.880,
     "Accuracy":          0.971,
     "Note":              "Wang et al. 2020"},
    {"Method":            "WaPIRL (self-supervised)",
     "Backbone":          "ResNet-18",
     "Macro F1":          0.891,
     "Accuracy":          0.974,
     "Note":              "Nakazawa 2021"},
    {"Method":            "MCNN-WaferNet",
     "Backbone":          "Multi-scale CNN",
     "Macro F1":          0.912,
     "Accuracy":          0.978,
     "Note":              "Shim et al. 2022"},
]

# ─────────────────────────────────────────────────────────────────────────────

def _bar(val: float, width: int = 20) -> str:
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled)


def _col(val: float, hi: float = 0.95, lo: float = 0.85) -> str:
    if val >= hi:
        return f"\033[92m{val:.3f}\033[0m"   # green
    if val >= lo:
        return f"\033[93m{val:.3f}\033[0m"   # yellow
    return f"\033[91m{val:.3f}\033[0m"        # red


@torch.no_grad()
def run_evaluation():
    setup_logging()
    cfg = parse_args_to_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("evaluate_testset", device=str(device))

    # ── load data ──────────────────────────────────────────────────────────
    _, _, test_loader, _ = build_classifier_loaders(cfg)
    class_names = cfg.wm811k.class_names

    # ── load model ─────────────────────────────────────────────────────────
    ckpt_path = Path(cfg.serving.model_path)
    if not ckpt_path.exists():
        ckpt_path = Path("artifacts/classifier_best.pt")
    log.info("loading_checkpoint", path=str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = WaferClassifier(
        backbone_name=cfg.classifier.backbone,
        num_classes=cfg.wm811k.num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("model_loaded", epoch=ckpt.get("epoch", "?"), val_f1=f"{ckpt.get('metric', 0):.4f}")

    # ── inference ──────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # ── aggregate metrics ──────────────────────────────────────────────────
    acc       = accuracy_score(labels, preds)
    f1_macro  = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weight = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_per    = f1_score(labels, preds, average=None,       zero_division=0)

    # ── PRINT RESULTS ──────────────────────────────────────────────────────
    sep = "─" * 72

    print(f"\n{'═'*72}")
    print(f"  WaferGuard — Test Set Results  ({len(labels):,} samples)")
    print(f"{'═'*72}")
    print(f"  Accuracy        : {_col(acc)}   {_bar(acc)}")
    print(f"  Macro F1        : {_col(f1_macro)}   {_bar(f1_macro)}")
    print(f"  Weighted F1     : {_col(f1_weight)}   {_bar(f1_weight)}")
    print(f"{sep}")

    # per-class
    print(f"  {'Class':<14}  {'F1':>6}  {'bar':^22}  Count")
    print(f"  {'-'*14}  {'-'*6}  {'-'*22}  -----")
    for i, name in enumerate(class_names):
        count = int((labels == i).sum())
        if count == 0:
            continue
        f1 = f1_per[i] if i < len(f1_per) else 0.0
        bar = _bar(f1, 22)
        print(f"  {name:<14}  {_col(f1, 0.93, 0.80):>6}  {bar}  {count:>5}")

    # ── LEADERBOARD ────────────────────────────────────────────────────────
    waferguard_row = {
        "Method":    f"WaferGuard (ours)",
        "Backbone":  f"ResNet-34 + Focal",
        "Macro F1":  f1_macro,
        "Accuracy":  acc,
        "Note":      "this run",
    }
    all_rows = BASELINES + [waferguard_row]
    all_rows.sort(key=lambda r: r["Macro F1"], reverse=True)

    print(f"\n{'═'*72}")
    print(f"  WM-811K Leaderboard (Defect Pattern Classification)")
    print(f"{'═'*72}")
    print(f"  {'#':>2}  {'Method':<30}  {'Backbone':<20}  {'F1':>5}  {'Acc':>5}")
    print(f"  {'-'*2}  {'-'*30}  {'-'*20}  {'-'*5}  {'-'*5}")
    for rank, row in enumerate(all_rows, 1):
        marker = " ◄" if row["Method"].startswith("WaferGuard") else ""
        method = row["Method"][:30]
        backbone = row["Backbone"][:20]
        f1v  = row["Macro F1"]
        accv = row["Accuracy"]
        f1s  = _col(f1v,  0.93, 0.80)
        accs = _col(accv, 0.97, 0.94)
        print(f"  {rank:>2}  {method:<30}  {backbone:<20}  {f1s:>5}  {accs:>5}{marker}")
    print(f"{'═'*72}\n")

    # ── confusion matrix (compact) ─────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    print(f"  Confusion matrix (rows=true, cols=pred):")
    header = "  " + "".join(f"{n[:4]:>6}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        label = f"{class_names[i][:4]:>6}"
        cells = "".join(
            f"\033[92m{v:>6}\033[0m" if j == i and v > 0 else
            f"\033[91m{v:>6}\033[0m" if j != i and v > 0 else
            f"{'0':>6}"
            for j, v in enumerate(row)
        )
        print(f"  {label}{cells}")
    print()


if __name__ == "__main__":
    run_evaluation()
