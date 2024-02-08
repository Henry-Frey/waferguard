"""Inference utilities — ONNX runtime and batch inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.logging import get_logger

log = get_logger(__name__)


class ONNXInferenceEngine:
    """ONNX Runtime-based inference for production deployment."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        log.info("onnx_model_loaded", path=str(model_path))

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run inference on a batch of images.

        Args:
            images: (B, C, H, W) float32 array

        Returns:
            logits: (B, num_classes) float32 array
        """
        outputs = self.session.run(None, {self.input_name: images})
        return outputs[0]
