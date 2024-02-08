"""FastAPI application for wafer defect detection."""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.models.classifier import WaferClassifier
from src.models.patchcore import PatchCore
from src.serving.schemas import (
    AnomalyResult, DefectClassification, HealthResponse, WaferResponse,
)
from src.utils.config import load_config
from src.utils.logging import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)

REQUEST_COUNT = Counter("wafer_requests_total", "Total inference requests")
LATENCY = Histogram("wafer_latency_ms", "Inference latency", buckets=[10, 25, 50, 100, 250, 500])
DEFECT_COUNT = Counter("wafer_defects_total", "Detected defects", ["defect_type"])

classifier = None
patchcore_model = None
cfg = None
device = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, patchcore_model, cfg, device
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.serving.mode in ("classifier", "dual"):
        ckpt = torch.load(cfg.serving.model_path, map_location=device, weights_only=False)
        classifier = WaferClassifier(
            backbone_name=cfg.classifier.backbone,
            num_classes=cfg.wm811k.num_classes,
        ).to(device)
        classifier.load_state_dict(ckpt["model"])
        classifier.eval()
        log.info("classifier_loaded")

    if cfg.serving.mode in ("anomaly", "dual"):
        patchcore_model = PatchCore(
            backbone_name=cfg.anomaly.backbone,
            layers=cfg.anomaly.layers,
            k_nearest=cfg.anomaly.k_nearest,
        )
        patchcore_model.load(cfg.serving.anomaly_bank_path)
        log.info("anomaly_model_loaded")

    yield
    log.info("server_shutdown")


app = FastAPI(
    title="WaferGuard API",
    version="1.0.0",
    description="Semiconductor wafer defect detection and anomaly localization",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        classifier_loaded=classifier is not None,
        anomaly_model_loaded=patchcore_model is not None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/predict", response_model=WaferResponse)
async def predict(file: UploadFile = File(...)):
    REQUEST_COUNT.inc()
    t0 = time.perf_counter()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    result = WaferResponse(
        inference_time_ms=0, mode=cfg.serving.mode
    )

    img_tensor = _preprocess(image)

    # classification
    if classifier is not None:
        with torch.no_grad():
            logits = classifier(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            pred_name = cfg.wm811k.class_names[pred_idx]

        DEFECT_COUNT.labels(defect_type=pred_name).inc()
        result.classification = DefectClassification(
            predicted_class=pred_name,
            confidence=probs[pred_idx].item(),
            all_probabilities={
                name: probs[i].item()
                for i, name in enumerate(cfg.wm811k.class_names)
            },
        )

    # anomaly detection
    if patchcore_model is not None:
        scores, maps = patchcore_model.predict(img_tensor)
        score = scores[0].item()
        result.anomaly = AnomalyResult(
            anomaly_score=score,
            is_anomalous=score > cfg.anomaly.threshold_percentile,
            threshold=cfg.anomaly.threshold_percentile,
        )

    result.inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


def _preprocess(image: Image.Image) -> torch.Tensor:
    """Convert a color-coded wafer map PNG to the 3-channel binary encoding.

    Training used:
      ch0 = tested die  (wm > 0)
      ch1 = passing die (wm == 1)  -> encoded as green in our test PNGs
      ch2 = failing die (wm == 2)  -> encoded as red in our test PNGs

    We decode the uploaded RGB image back to that binary representation so the
    model receives the same format it was trained on.
    """
    img = image.resize((cfg.data.img_size, cfg.data.img_size), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_pass = (G > 0.4) & (R < 0.6)   # greenish → pass die
    is_fail = (R > 0.4) & (G < 0.5)   # reddish  → fail die
    is_tested = is_pass | is_fail

    ch = np.stack([
        is_tested.astype(np.float32),
        is_pass.astype(np.float32),
        is_fail.astype(np.float32),
    ], axis=0)  # (3, H, W)

    return torch.from_numpy(ch).unsqueeze(0).to(device)
