"""WM-811K wafer map dataset — loading, preprocessing, and PyTorch Dataset.

The WM-811K dataset (Wu et al., 2014) contains 811,457 wafer bin maps from a
real semiconductor fab.  Each map is a 2D array where 0=untested, 1=pass, 2=fail.
Only 172,950 maps are expert-labeled with one of 8 defect patterns + None.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compatibility shim for old pandas pickles
# ---------------------------------------------------------------------------

def _new_Index_compat(cls, d):
    """Replacement for pandas.core.indexes._new_Index removed in pandas 2.x."""
    d.pop("fastpath", None)
    return cls.__new__(cls, **d)


class _WM811KUnpickler(pickle.Unpickler):
    """Custom unpickler for LSWMD.pkl saved with pandas ~0.20 / Python 2."""

    def __init__(self, f):
        super().__init__(f, encoding="latin1", errors="replace")

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

    _CLASS_MAP: dict[tuple[str, str], tuple[str, str]] = {
        ("pandas.core.indexes", "Index"): ("pandas.core.indexes.base", "Index"),
    }

    def find_class(self, module: str, name: str):
        if name == "_new_Index":
            return _new_Index_compat
        module = self._MODULE_MAP.get(module, module)
        module, name = self._CLASS_MAP.get((module, name), (module, name))
        return super().find_class(module, name)


LABEL_MAP = {
    "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
    "Loc": 4, "Near-full": 5, "Random": 6, "Scratch": 7, "none": 8,
}


# ---------------------------------------------------------------------------
# Raw loading helpers
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

    wafer_maps: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        label_str = row.get("failureType")
        if not isinstance(label_str, str) and hasattr(label_str, "__len__"):
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


def denoise_wafer_map(
    wm: np.ndarray, window: int = 3, min_neighbors: int = 3
) -> np.ndarray:
    """Remove isolated noise dies by neighbor voting.

    A fail die (value=2) is kept only if it has >= min_neighbors
    other fail dies in the surrounding window.
    """
    fail_mask = (wm == 2).astype(np.uint8)
    neighbor_count = np.zeros_like(fail_mask, dtype=np.int32)

    for dy in range(-window // 2, window // 2 + 1):
        for dx in range(-window // 2, window // 2 + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(fail_mask, dy, axis=0), dx, axis=1)
            neighbor_count += shifted

    cleaned = wm.copy()
    noise_mask = (fail_mask == 1) & (neighbor_count < min_neighbors)
    cleaned[noise_mask] = 1
    return cleaned


def preprocess_wm811k(
    raw_path: str | Path,
    img_size: int = 96,
    noise_filter: bool = True,
    window_size: int = 3,
    min_neighbors: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing: load → denoise → resize.

    Returns:
        wafer_maps: (N, H, W) uint8 array
        labels: (N,) int64 array
    """
    from PIL import Image

    wafer_maps_raw, labels = load_wm811k_raw(raw_path)

    processed = []
    for wm in wafer_maps_raw:
        if noise_filter:
            wm = denoise_wafer_map(wm, window_size, min_neighbors)
        wm_img = Image.fromarray(wm.astype(np.uint8), mode="L")
        wm_img = wm_img.resize((img_size, img_size), Image.NEAREST)
        processed.append(np.array(wm_img))

    return np.stack(processed), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WM811KDataset(Dataset):
    """Preprocessed WM-811K wafer map dataset for classification.

    Encodes each wafer map as a 3-channel float32 image:
        ch0 = die mask  (1 where a die was tested)
        ch1 = pass map  (1 where die passed)
        ch2 = fail map  (1 where die failed)
    """

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
        label = int(self.labels[idx])

        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        h, w = wm.shape
        y_off = max(0, (self.img_size - h) // 2)
        x_off = max(0, (self.img_size - w) // 2)
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
