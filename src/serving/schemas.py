"""API schemas for wafer defect detection."""

from __future__ import annotations

from pydantic import BaseModel


class DefectClassification(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict[str, float]


class AnomalyResult(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    threshold: float


class WaferResponse(BaseModel):
    classification: DefectClassification | None = None
    anomaly: AnomalyResult | None = None
    inference_time_ms: float
    mode: str  # classifier | anomaly | dual


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    anomaly_model_loaded: bool
    device: str
