"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skip(reason="Requires trained model artifacts")
def test_health_endpoint():
    from src.serving.app import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.skip(reason="Requires trained model artifacts")
def test_predict_endpoint_invalid_file():
    from src.serving.app import app
    client = TestClient(app)
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
