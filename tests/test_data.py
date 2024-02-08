import numpy as np
from src.data.wm811k import denoise_wafer_map, WM811KDataset


def test_denoise_removes_isolated_fails():
    wm = np.ones((10, 10), dtype=np.uint8)
    wm[5, 5] = 2  # single isolated fail
    cleaned = denoise_wafer_map(wm, window=3, min_neighbors=2)
    assert cleaned[5, 5] == 1  # should be removed


def test_denoise_preserves_clusters():
    wm = np.ones((10, 10), dtype=np.uint8)
    wm[4:7, 4:7] = 2  # 3x3 cluster of fails
    cleaned = denoise_wafer_map(wm, window=3, min_neighbors=2)
    assert cleaned[5, 5] == 2  # center should survive


def test_dataset_output_shape(dummy_wafer_maps):
    maps, labels = dummy_wafer_maps
    ds = WM811KDataset(maps, labels, img_size=96)
    sample = ds[0]
    assert sample["image"].shape == (3, 96, 96)
    assert sample["label"].ndim == 0
