"""Wafer-specific augmentation pipelines using albumentations."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_classifier_transforms(img_size: int, is_train: bool = True) -> A.Compose:
    """Build albumentations pipeline for wafer map classification.

    Key choices:
    - Full 360° rotation (p=1.0): defect labels are rotation-invariant.
    - H/V flips (p=0.5): no preferred orientation in wafer maps.
    - Elastic deform (p=0.2): simulates die-placement variation.
    - Affine (p=0.3): handles wafer centering imprecision.
    """
    if is_train:
        return A.Compose([
            A.Rotate(limit=180, p=1.0, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=0,
                p=0.3,
            ),
            A.ElasticTransform(alpha=20.0, sigma=5.0, p=0.2),
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])


def build_anomaly_transforms(img_size: int, is_train: bool = True) -> A.Compose:
    """Minimal transforms for PatchCore anomaly detection."""
    transforms = [A.Resize(img_size, img_size)]
    if is_train:
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return A.Compose(transforms)
