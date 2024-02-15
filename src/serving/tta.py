"""Test-Time Augmentation (TTA) for wafer defect classification.

8-fold TTA averages softmax probabilities across all rotations and flips:
    4 rotations (0°, 90°, 180°, 270°) × 2 flip states = 8 views

This is a free inference-time improvement — no retraining required.
Expected gain: +0.5–1.0% macro F1 on WM-811K.

Selective TTA applies augmentation only to low-confidence predictions
(max softmax < threshold), reducing the 8× inference cost substantially
when most samples are easy cases.

Usage:
    from src.serving.tta import tta_predict, selective_tta_predict

    # Full 8-fold TTA
    probs = tta_predict(model, images, device)
    preds = probs.argmax(dim=1)

    # Selective TTA (only augment uncertain samples)
    probs = selective_tta_predict(model, images, device, confidence_threshold=0.9)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tta_predict(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device | str,
    n_augments: int = 8,
) -> torch.Tensor:
    """8-fold TTA: 4 rotations × 2 flip states → averaged softmax.

    Args:
        model: trained WaferClassifier or WaferClassifierFPN (in eval mode)
        images: (B, C, H, W) tensor on CPU or GPU
        device: target device
        n_augments: number of augmentation views to use (max 8; fewer = faster)

    Returns:
        (B, num_classes) averaged probability tensor on CPU
    """
    model.eval()
    images = images.to(device)
    all_probs: list[torch.Tensor] = []

    # Generate 8 views: 4 rotations × (original, horizontal flip)
    for k in range(4):
        rotated = torch.rot90(images, k=k, dims=[2, 3])
        with torch.no_grad():
            all_probs.append(F.softmax(model(rotated), dim=1).cpu())
        if len(all_probs) >= n_augments:
            break
        flipped = torch.flip(rotated, dims=[3])
        with torch.no_grad():
            all_probs.append(F.softmax(model(flipped), dim=1).cpu())
        if len(all_probs) >= n_augments:
            break

    return torch.stack(all_probs[:n_augments]).mean(dim=0)


def selective_tta_predict(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device | str,
    confidence_threshold: float = 0.9,
    n_augments: int = 8,
) -> torch.Tensor:
    """Selective TTA: apply 8-fold TTA only to uncertain predictions.

    Samples whose max softmax probability exceeds `confidence_threshold`
    are classified with a single forward pass (fast path).  Only uncertain
    samples pay the 8× cost.  Reduces average inference time by ~70% when
    most samples are confident, with negligible accuracy difference.

    Args:
        model: trained classifier in eval mode
        images: (B, C, H, W) input batch
        device: target device
        confidence_threshold: max-prob threshold above which TTA is skipped
        n_augments: TTA views for uncertain samples

    Returns:
        (B, num_classes) probability tensor on CPU
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        base_probs = F.softmax(model(images), dim=1).cpu()

    max_probs = base_probs.max(dim=1).values
    uncertain_mask = max_probs < confidence_threshold

    if not uncertain_mask.any():
        return base_probs

    uncertain_imgs = images[uncertain_mask]
    uncertain_probs = tta_predict(model, uncertain_imgs, device, n_augments=n_augments)
    base_probs[uncertain_mask] = uncertain_probs
    return base_probs
