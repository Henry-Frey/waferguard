"""WM-811K wafer map dataset — loading, preprocessing, and splitting.

The WM-811K dataset (Wu et al., 2014) contains 811,457 wafer bin maps
from a real semiconductor fab. Each wafer map is a 2D array where:
  0 = untested die, 1 = pass, 2 = fail

Only 172,950 maps are expert-labeled with one of 8 defect patterns + None.
This module handles the full pipeline from raw .pkl to training-ready tensors.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label, rotate as nd_rotate
from torch.utils.data import Dataset

from src.utils.logging import get_logger

log = get_logger(__name__)


def _new_Index_compat(cls, d):
    """Replacement for pandas.core.indexes._new_Index removed in pandas 2.x."""
    d.pop("fastpath", None)
    return cls.__new__(cls, **d)


class _WM811KUnpickler(pickle.Unpickler):
    """Custom unpickler for LSWMD.pkl saved with pandas ~0.20 / Python 2.

    Two categories of fixes:
    1. encoding='latin1'  — Python 2 pickled byte strings as raw bytes; latin1
       is the correct codec to round-trip them into Python 3 str/bytes.
    2. find_class() remapping — the pickle stores old pandas internal module
       paths that no longer exist in pandas 2.x.

    Known remaps needed:
      pandas.indexes.*           → pandas.core.indexes.*
      pandas.core.indexes.Index  → pandas.core.indexes.base.Index
      _new_Index (any module)    → local compat shim
    """

    def __init__(self, f):
        super().__init__(f, encoding="latin1", errors="replace")

    # old module → new module
    _MODULE_MAP: dict[str, str] = {
        "pandas.indexes": "pandas.core.indexes",
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.range": "pandas.core.indexes.range",
        "pandas.indexes.multi": "pandas.core.indexes.multi",
        "pandas.indexes.frozen": "pandas.core.indexes.frozen",
        "pandas.indexes.numeric": "pandas.core.indexes.numeric",
        "pandas.indexes.category": "pandas.core.indexes.category",
        "pandas.indexes.datetimes": "pandas.core.indexes.datetimes",
        "pandas.indexes.period": "pandas.core.indexes.period",
        "pandas.indexes.timedeltas": "pandas.core.indexes.timedeltas",
    }

    # (old_module, name) → (new_module, name)  for classes that moved
    _CLASS_MAP: dict[tuple[str, str], tuple[str, str]] = {
        # Index was re-exported at top level but lives in .base
        ("pandas.core.indexes", "Index"): ("pandas.core.indexes.base", "Index"),
    }

    def find_class(self, module: str, name: str):
        # 1. Remap legacy _new_Index to our compat shim (no module needed)
        if name == "_new_Index":
            return _new_Index_compat

        # 2. Remap old module paths
        module = self._MODULE_MAP.get(module, module)

        # 3. Remap specific (module, class) pairs that moved
        module, name = self._CLASS_MAP.get((module, name), (module, name))

        return super().find_class(module, name)

LABEL_MAP = {
    "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
    "Loc": 4, "Near-full": 5, "Random": 6, "Scratch": 7, "none": 8,
}


# ---------------------------------------------------------------------------
# Engineered feature channels
# ---------------------------------------------------------------------------

def compute_distance_from_center(img_size: int) -> np.ndarray:
    """Normalized distance-from-center map for an (img_size × img_size) image.

    Value at center pixel = 0.0, value at corners = 1.0.  This channel makes
    the Loc/Edge-Loc radial position difference explicit and trivially learnable.

    Returns:
        (img_size, img_size) float32 array
    """
    ys = np.linspace(-1.0, 1.0, img_size, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, img_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    # divide by sqrt(2) so the corner distance is exactly 1.0
    return np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2.0)


def compute_radon_channel(
    fail_mask: np.ndarray,
    n_angles: int = 36,
) -> np.ndarray:
    """Radon transform sinogram of the fail-die mask, resized to (H, H).

    The Radon transform projects the fail pattern along multiple angles.
    Scratch defects are thin linear structures that produce sharp, narrow
    peaks in the sinogram at the scratch orientation angle — giving the
    network an orientation-explicit representation unavailable in the raw
    binary map.

    Implementation uses scipy.ndimage.rotate (already a dependency) to
    rotate the mask and accumulate column sums, avoiding the scikit-image
    dependency.  For 36 angles on a 128×128 mask this runs in ~2 ms/sample.

    Args:
        fail_mask: (H, W) binary float32 array of fail dies
        n_angles: number of projection angles in [0°, 180°)

    Returns:
        (H, H) float32 sinogram normalised to [0, 1]
    """
    from PIL import Image as PIL_Image

    h = fail_mask.shape[0]
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = np.zeros((h, n_angles), dtype=np.float32)

    for i, angle in enumerate(angles):
        rotated = nd_rotate(fail_mask, angle, reshape=False, order=1)
        projection = rotated.sum(axis=1)
        max_val = projection.max()
        if max_val > 0:
            sinogram[:, i] = projection / max_val

    # Resize sinogram (H, n_angles) → (H, H) to match image spatial dimensions
    sino_pil = PIL_Image.fromarray((sinogram * 255).clip(0, 255).astype(np.uint8))
    sino_pil = sino_pil.resize((h, h), PIL_Image.BILINEAR)
    return np.array(sino_pil, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class WM811KDataset(Dataset):
    """Preprocessed WM-811K wafer map dataset for classification."""

    def __init__(
        self,
        wafer_maps: np.ndarray,
        labels: np.ndarray,
        img_size: int = 96,
        transforms: Any = None,
    ):
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        wm = self.wafer_maps[idx].astype(np.float32)
        label = self.labels[idx]

        # convert wafer map to 3-channel image:
        # channel 0 = die mask (1 where tested), channel 1 = pass, channel 2 = fail
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        h, w = wm.shape
        # center the wafer in the image
        y_off = (self.img_size - h) // 2
        x_off = (self.img_size - w) // 2
        y_off = max(0, y_off)
        x_off = max(0, x_off)
        h_clip = min(h, self.img_size)
        w_clip = min(w, self.img_size)

        region = wm[:h_clip, :w_clip]
        img[y_off:y_off + h_clip, x_off:x_off + w_clip, 0] = (region > 0).astype(np.float32)
        img[y_off:y_off + h_clip, x_off:x_off + w_clip, 1] = (region == 1).astype(np.float32)
        img[y_off:y_off + h_clip, x_off:x_off + w_clip, 2] = (region == 2).astype(np.float32)

        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


class WM811KDatasetV2(Dataset):
    """Extended WM-811K dataset with engineered feature channels.

    Adds up to two extra channels beyond the standard 3-channel encoding:
      ch3 = Radon sinogram of the fail-die mask (orientation signal for Scratch)
      ch4 = Normalized distance-from-center map (radial position for Loc/Edge-Loc)

    Total input: 5-channel tensor for use with WaferClassifierFPN.

    The Radon channel is computed per-sample at load time (~2 ms at 128×128
    with n_angles=36).  The distance channel is pre-computed once and shared.

    Args:
        wafer_maps: (N, H, W) array of preprocessed wafer maps (values 0,1,2)
        labels: (N,) integer label array
        img_size: spatial resolution of the input images
        transforms: albumentations Compose pipeline (must handle C-channel input)
        use_radon: include Radon sinogram channel
        use_distance: include distance-from-center channel
        radon_angles: number of projection angles for the Radon transform
    """

    def __init__(
        self,
        wafer_maps: np.ndarray,
        labels: np.ndarray,
        img_size: int = 96,
        transforms: Any = None,
        use_radon: bool = True,
        use_distance: bool = True,
        radon_angles: int = 36,
    ):
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.img_size = img_size
        self.transforms = transforms
        self.use_radon = use_radon
        self.use_distance = use_distance
        self.radon_angles = radon_angles
        # Pre-compute the distance map (same for every sample)
        self._dist_map = compute_distance_from_center(img_size) if use_distance else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        wm = self.wafer_maps[idx]   # (H, W), dtype uint8, values 0/1/2
        label = self.labels[idx]

        # --- 3-channel encoding (identical to WM811KDataset) ---
        wm_f = wm.astype(np.float32)
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        h, w = wm_f.shape
        y_off = max(0, (self.img_size - h) // 2)
        x_off = max(0, (self.img_size - w) // 2)
        h_clip = min(h, self.img_size)
        w_clip = min(w, self.img_size)
        region = wm_f[:h_clip, :w_clip]
        img[y_off:y_off + h_clip, x_off:x_off + w_clip, 0] = (region > 0)
        img[y_off:y_off + h_clip, x_off:x_off + w_clip, 1] = (region == 1)
        img[y_off:y_off + h_clip, x_off:x_off + w_clip, 2] = (region == 2)

        # --- engineered channels ---
        extra = []

        if self.use_radon:
            # Radon acts on the fail channel of the already-placed image
            fail_mask = img[:, :, 2]
            radon_ch = compute_radon_channel(fail_mask, self.radon_angles)
            extra.append(radon_ch[:, :, None])

        if self.use_distance:
            extra.append(self._dist_map[:, :, None])

        if extra:
            img = np.concatenate([img] + extra, axis=2)

        # --- transforms (albumentations handles arbitrary channel count) ---
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_wm811k_raw(path: str | Path) -> tuple[list[np.ndarray], list[int]]:
    """Load raw WM-811K pickle and extract labeled samples.

    Returns:
        wafer_maps: list of 2D numpy arrays (variable size)
        labels: list of integer labels
    """
    path = Path(path)
    pkl_file = path / "LSWMD.pkl" if path.is_dir() else path

    log.info("loading_wm811k", file=str(pkl_file))
    with open(pkl_file, "rb") as f:
        df = _WM811KUnpickler(f).load()

    wafer_maps = []
    labels = []

    for _, row in df.iterrows():
        label_str = row.get("failureType")
        if not isinstance(label_str, str) and hasattr(label_str, "__len__"):
            # some entries have nested arrays
            if len(label_str) == 0:
                continue
            label_str = label_str[0] if hasattr(label_str[0], "__len__") else label_str
            if hasattr(label_str, "__len__") and len(label_str) > 0:
                label_str = label_str[0]

        if not isinstance(label_str, str) or label_str not in LABEL_MAP:
            continue

        wm = np.array(row["waferMap"])
        if wm.ndim != 2 or wm.size == 0:
            continue

        wafer_maps.append(wm)
        labels.append(LABEL_MAP[label_str])

    log.info("wm811k_loaded", total_labeled=len(labels))
    return wafer_maps, labels


def load_wm811k_unlabeled(path: str | Path) -> list[np.ndarray]:
    """Load the ~638k unlabeled wafer maps for semi-supervised learning.

    Returns:
        list of 2D numpy arrays (variable size, values 0/1/2)
    """
    path = Path(path)
    pkl_file = path / "LSWMD.pkl" if path.is_dir() else path

    log.info("loading_wm811k_unlabeled", file=str(pkl_file))
    with open(pkl_file, "rb") as f:
        df = _WM811KUnpickler(f).load()

    unlabeled = []
    for _, row in df.iterrows():
        label_str = row.get("failureType")
        # keep entries that are NOT in the label map (unlabeled or empty)
        if isinstance(label_str, str) and label_str in LABEL_MAP:
            continue
        wm = np.array(row["waferMap"])
        if wm.ndim == 2 and wm.size > 0:
            unlabeled.append(wm)

    log.info("wm811k_unlabeled_loaded", total_unlabeled=len(unlabeled))
    return unlabeled


def denoise_wafer_map(
    wm: np.ndarray, window: int = 3, min_neighbors: int = 3
) -> np.ndarray:
    """Remove scattered noise from wafer map using neighbor voting.

    A fail die (value=2) is kept only if it has >= min_neighbors
    other fail dies within the local window. This removes isolated
    noise points while preserving clustered defect patterns.
    """
    fail_mask = (wm == 2).astype(np.uint8)
    neighbor_count = np.zeros_like(fail_mask, dtype=np.int32)

    # count fail neighbors (excluding self)
    for dy in range(-window // 2, window // 2 + 1):
        for dx in range(-window // 2, window // 2 + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(fail_mask, dy, axis=0), dx, axis=1)
            neighbor_count += shifted

    cleaned = wm.copy()
    noise_mask = (fail_mask == 1) & (neighbor_count < min_neighbors)
    cleaned[noise_mask] = 1  # flip noisy fails back to pass
    return cleaned


def preprocess_wm811k(
    raw_path: str | Path,
    img_size: int = 96,
    noise_filter: bool = True,
    window_size: int = 3,
    min_neighbors: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: load → denoise → resize → normalize.

    Returns:
        wafer_maps: (N, H, W) array of preprocessed wafer maps
        labels: (N,) array of integer labels
    """
    wafer_maps_raw, labels = load_wm811k_raw(raw_path)

    processed = []
    for wm in wafer_maps_raw:
        if noise_filter:
            wm = denoise_wafer_map(wm, window_size, min_neighbors)
        # resize to fixed dimensions via nearest interpolation
        from PIL import Image
        wm_img = Image.fromarray(wm.astype(np.uint8), mode="L")
        wm_img = wm_img.resize((img_size, img_size), Image.NEAREST)
        processed.append(np.array(wm_img))

    return np.stack(processed), np.array(labels)
