"""Export trained WaferClassifier to ONNX format for production deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.classifier import WaferClassifier
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def export_classifier_to_onnx(
    checkpoint_path: str,
    output_path: str,
    img_size: int = 96,
    opset_version: int = 17,
) -> None:
    setup_logging()
    cfg = load_config()

    model = WaferClassifier(
        backbone_name=cfg.classifier.backbone,
        num_classes=cfg.wm811k.num_classes,
        pretrained=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    log.info("onnx_exported", path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export WaferClassifier to ONNX")
    parser.add_argument("--checkpoint", default="artifacts/classifier_best.pt")
    parser.add_argument("--output", default="artifacts/classifier.onnx")
    parser.add_argument("--img-size", type=int, default=96)
    args = parser.parse_args()

    export_classifier_to_onnx(args.checkpoint, args.output, args.img_size)
