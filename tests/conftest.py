import pytest
import torch
import numpy as np
from omegaconf import OmegaConf


@pytest.fixture
def dummy_config():
    return OmegaConf.create({
        "seed": 42,
        "data": {"root": "./data", "img_size": 96, "num_workers": 0, "pin_memory": False},
        "wm811k": {
            "path": "./data/wm811k",
            "num_classes": 9,
            "class_names": [
                "Center", "Donut", "Edge-Loc", "Edge-Ring",
                "Loc", "Near-Full", "Random", "Scratch", "None",
            ],
        },
        "classifier": {
            "backbone": "resnet18", "pretrained": False,
            "batch_size": 4, "epochs": 1,
        },
        "anomaly": {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "img_size": 224, "coreset_ratio": 0.1, "k_nearest": 3,
        },
    })


@pytest.fixture
def dummy_wafer_batch():
    return {
        "image": torch.randn(4, 3, 96, 96),
        "label": torch.tensor([0, 1, 3, 8]),
    }


@pytest.fixture
def dummy_wafer_maps():
    maps = np.random.choice([0, 1, 2], size=(20, 52, 52), p=[0.2, 0.6, 0.2])
    labels = np.random.randint(0, 9, size=20)
    return maps, labels
