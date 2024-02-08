import torch
from src.models.classifier import WaferClassifier


def test_classifier_forward(dummy_config):
    model = WaferClassifier(
        backbone_name="resnet18", num_classes=9, pretrained=False
    )
    x = torch.randn(2, 3, 96, 96)
    out = model(x)
    assert out.shape == (2, 9)


def test_classifier_feature_extraction(dummy_config):
    model = WaferClassifier(
        backbone_name="resnet18", num_classes=9, pretrained=False
    )
    x = torch.randn(2, 3, 96, 96)
    feat = model.extract_features(x)
    assert feat.shape[0] == 2
    assert feat.ndim == 2
