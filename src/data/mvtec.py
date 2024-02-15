"""MVTec AD dataset loader (optional benchmark for anomaly detection)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """MVTec Anomaly Detection dataset loader.

    Used to validate the anomaly detection branch on a well-known
    industrial benchmark before applying to wafer data.
    """

    CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        img_size: int = 224,
        transforms: Any = None,
    ):
        self.root = Path(root) / category
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples = []
        split_dir = self.root / self.split
        if not split_dir.exists():
            return samples

        for class_dir in sorted(split_dir.iterdir()):
            label = 0 if class_dir.name == "good" else 1
            for img_path in sorted(class_dir.glob("*.png")):
                samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_arr = np.array(image, dtype=np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=img_arr)
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        return {"image": img_tensor, "label": torch.tensor(label, dtype=torch.long)}
