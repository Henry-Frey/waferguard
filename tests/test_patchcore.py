import torch
import numpy as np
from src.models.memory_bank import MemoryBank
from src.models.feature_extractor import FeatureExtractor


def test_memory_bank():
    bank = MemoryBank(coreset_ratio=0.5)
    features = torch.randn(100, 64)
    bank.fit(features)
    assert bank.features is not None
    assert bank.features.shape[0] == 50  # 50% coreset


def test_memory_bank_scoring():
    bank = MemoryBank(coreset_ratio=1.0)
    bank.fit(torch.randn(50, 64))
    scores = bank.score(torch.randn(10, 64), k=3)
    assert scores.shape == (10,)
    assert (scores >= 0).all()


def test_feature_extractor():
    extractor = FeatureExtractor(backbone_name="wide_resnet50_2", layers=["layer2", "layer3"])
    x = torch.randn(1, 3, 224, 224)
    feats = extractor(x)
    assert "layer2" in feats
    assert "layer3" in feats


def test_patch_features():
    extractor = FeatureExtractor(backbone_name="wide_resnet50_2", layers=["layer2", "layer3"])
    x = torch.randn(2, 3, 224, 224)
    patches = extractor.get_patch_features(x, target_size=28)
    assert patches.ndim == 2
    assert patches.shape[0] == 2 * 28 * 28
