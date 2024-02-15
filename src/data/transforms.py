"""Wafer-specific augmentation pipelines."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_classifier_transforms(
    img_size: int,
    is_train: bool = True,
    extra_channels: int = 0,
) -> A.Compose:
    """Augmentations for wafer map classification.

    Key design choices validated by the WM-811K literature:

    Rotation (continuous, p=1.0):
        Defect labels are rotation-invariant by convention — Edge-Ring looks
        the same at any angle.  Using full 0–360° rotation at p=1.0 doubles
        effective training diversity vs the common 90°-only / p=0.5 approach.
        Especially critical for Scratch (thin linear features at arbitrary angles).

    Flip (p=0.5 each axis):
        Wafer maps have no preferred orientation; both horizontal and vertical
        flips are valid data augmentations for all defect types.

    Elastic deformation (p=0.2):
        Small elastic warps simulate die-placement variation and manufacturing
        tolerances without destroying the global defect geometry.

    Affine (p=0.3):
        Small translations/scales handle wafer centering imprecision.

    The extra_channels parameter is informational only — albumentations
    handles arbitrary channel counts transparently.

    Args:
        img_size: target spatial resolution
        is_train: True for augmented training transforms, False for eval
        extra_channels: number of extra channels beyond 3 (unused, for docs)
    """
    if is_train:
        return A.Compose([
            # Full 360° rotation at p=1.0 — most impactful single augmentation
            A.Rotate(limit=180, p=1.0, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=0,
                p=0.3,
            ),
            # Elastic deformation: small warps simulate die-placement variation
            A.ElasticTransform(
                alpha=20.0,
                sigma=5.0,
                p=0.2,
            ),
            # Note: GaussNoise omitted — albumentations' cv2 backend has a dtype
            # mismatch on float32 multi-channel images in newer versions, and
            # rotation + elastic deform + CutMix provide ample regularization.
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])


def build_anomaly_transforms(img_size: int, is_train: bool = True) -> A.Compose:
    """Transforms for anomaly detection branch.

    Minimal augmentation — PatchCore relies on consistent feature extraction.
    """
    transforms = [A.Resize(img_size, img_size)]
    if is_train:
        transforms.append(A.HorizontalFlip(p=0.5))
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return A.Compose(transforms)
