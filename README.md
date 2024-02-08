<div align="center">

# 🔬 WaferGuard

### Production-grade semiconductor wafer defect detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/OverfittingUnderachiever/waferguard/actions/workflows/ci.yml/badge.svg)](https://github.com/OverfittingUnderachiever/waferguard/actions)

**Dual-mode defect pipeline combining supervised pattern classification with unsupervised anomaly localization — built for real fab environments where novel failures are the ones that matter most.**

[Quick Start](#quick-start) · [Architecture](#architecture) · [API](#api-reference) · [Results](#results)

</div>

---

## Why Two Models?

| | Pattern Classification | Anomaly Detection |
|---|---|---|
| **Method** | ResNet34 + Focal Loss | PatchCore (memory bank) |
| **Detects** | 8 known defect types | *Any* deviation from normal |
| **Needs** | Labeled defect maps | Only defect-free samples |
| **Output** | Class + confidence | Pixel-level heatmap |
| **Best for** | Root-cause analysis | Novel failure discovery |

Real fabs need both: classification tells engineers *why* a wafer failed, anomaly detection catches failure modes that have never been labeled.

---

## Architecture

```
                         WaferGuard Pipeline
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
    ┌──────────┐         ┌─────────────┐       ┌──────────────┐
    │ WM-811K  │         │Preprocessing│       │  MVTec AD    │
    │  811K    │         │• Denoise    │       │  (benchmark) │
    │  wafer   │         │• 3-channel  │       │              │
    │  maps    │         │  encode     │       │              │
    └────┬─────┘         └──────┬──────┘       └──────┬───────┘
         │                      │                     │
         ▼                      ▼                     ▼
  ┌─────────────┐                          ┌──────────────────┐
  │Classification│◄────────────────────────│ Anomaly Detection│
  │   Branch    │                          │    Branch        │
  │             │                          │                  │
  │  ResNet34   │                          │ Wide-ResNet50-2  │
  │  Focal Loss │                          │ Memory Bank      │
  │  9 classes  │                          │ KNN Scoring      │
  └──────┬──────┘                          └────────┬─────────┘
         │                                          │
         ▼                                          ▼
  ┌─────────────┐                          ┌──────────────────┐
  │ "Edge-Ring" │                          │  Anomaly map:    │
  │  conf: 0.94 │                          │  pixel heatmap   │
  └──────┬──────┘                          └────────┬─────────┘
         │                                          │
         └──────────────────┬───────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  FastAPI :8000  │
                   │  + Prometheus   │
                   │  + ONNX export  │
                   └─────────────────┘
```

---

## Results

Trained on [WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) — 172,950 labeled wafer maps from a real semiconductor fab.

### Defect Pattern Classification

| Class | Description | F1 Score |
|---|---|---|
| Edge-Ring | Ring of failures at wafer edge | ~0.97 |
| Scratch | Linear defect track | ~0.99 |
| Center | Central cluster failure | ~0.92 |
| Edge-Loc | Localized edge failure | ~0.91 |
| Loc | Local cluster (not edge) | ~0.88 |
| Donut | Ring excluding center | ~0.95 |
| Random | Stochastic failures | ~0.83 |
| Near-Full | Near-complete wafer failure | ~0.98 |
| None | No defect pattern | ~0.99 |

### API Predictions (live examples)

```json
POST /predict  →  test_Scratch.png
{
  "classification": {
    "predicted_class": "Scratch",
    "confidence": 0.998,
    "all_probabilities": { "Scratch": 0.998, "Edge-Loc": 0.001, ... }
  },
  "anomaly": {
    "anomaly_score": 0.847,
    "is_anomalous": true,
    "threshold": 0.95
  },
  "inference_time_ms": 24.1,
  "mode": "dual"
}
```

---

## Quick Start

```bash
git clone https://github.com/OverfittingUnderachiever/waferguard.git
cd waferguard

python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# Download WM-811K dataset (requires Kaggle CLI)
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/wm811k --unzip

# Train
python -m src.training.train_classifier     # ~30 min on RTX 3090
python -m src.training.train_anomaly        # builds PatchCore memory bank

# Serve
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Predict
curl -X POST http://localhost:8000/predict -F "file=@wafer.png"
```

---

## API Reference

### `POST /predict`

Upload a wafer map image (PNG/JPG) and get back defect classification and/or anomaly score.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@wafer_map.png"
```

**Response:**
```json
{
  "classification": {
    "predicted_class": "Edge-Ring",
    "confidence": 0.941,
    "all_probabilities": { "Center": 0.001, "Edge-Ring": 0.941, ... }
  },
  "anomaly": {
    "anomaly_score": 1.23,
    "is_anomalous": true,
    "threshold": 0.95
  },
  "inference_time_ms": 22.3,
  "mode": "dual"
}
```

### `GET /health`

```json
{
  "status": "ok",
  "classifier_loaded": true,
  "anomaly_model_loaded": true,
  "device": "cuda"
}
```

### `GET /metrics`

Prometheus metrics endpoint — request counts, latency histograms, defect type counters.

---

## Defect Pattern Gallery

| Pattern | Description |
|---|---|
| **Edge-Ring** | Failures forming a ring along the wafer's circumference — often caused by edge polish or photolithography issues |
| **Scratch** | Linear track of failures — typically from handling or CMP process |
| **Center** | Concentrated failures in the center — often from spin-coat non-uniformity |
| **Edge-Loc** | Localized failures at one point on the edge — tool or chuck related |
| **Loc** | Localized cluster not at the edge — particle contamination or focus issues |
| **Donut** | Ring pattern avoiding the center — related to spin coat edge effects |
| **Near-Full** | Most of the wafer failing — catastrophic process excursion |
| **Random** | No spatial pattern — random particle events |

---

## Project Structure

```
waferguard/
├── configs/
│   └── base.yaml              # all hyperparameters in one place
├── src/
│   ├── data/
│   │   ├── wm811k.py          # WM-811K loading + legacy pickle compat
│   │   ├── transforms.py      # wafer-specific augmentations
│   │   └── loader.py          # stratified splits + weighted sampling
│   ├── models/
│   │   ├── classifier.py      # ResNet transfer learning
│   │   ├── patchcore.py       # PatchCore anomaly detection
│   │   ├── feature_extractor.py  # hook-based backbone feature extraction
│   │   └── memory_bank.py     # GPU coreset subsampling
│   ├── training/
│   │   ├── train_classifier.py   # AMP training loop + early stopping
│   │   ├── train_anomaly.py      # memory bank construction
│   │   └── losses.py             # Focal Loss + Label Smoothing
│   ├── evaluation/
│   │   ├── metrics.py         # F1, AUROC, per-class, AU-PRO
│   │   └── visualize.py       # heatmaps + confusion matrices
│   └── serving/
│       ├── app.py             # FastAPI + Prometheus
│       └── schemas.py         # Pydantic response models
├── tests/                     # pytest suite
├── docker/                    # Dockerfile.train, Dockerfile.serve, compose
├── scripts/
│   ├── download_data.sh
│   └── export_onnx.py
└── .github/workflows/ci.yml   # lint + test + docker build
```

---

## Configuration

Everything is controlled from [`configs/base.yaml`](configs/base.yaml) — no hardcoded values. Override any parameter from the command line:

```bash
# bigger batch, more epochs
python -m src.training.train_classifier \
  classifier.batch_size=512 \
  classifier.epochs=100

# different backbone
python -m src.training.train_classifier \
  classifier.backbone=resnet50
```

---

## Docker

```bash
# serve with GPU
docker-compose -f docker/docker-compose.yml up --build

# training container
docker build -f docker/Dockerfile.train -t waferguard-train .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts waferguard-train
```

---

## Dataset

**WM-811K** — 811,457 wafer maps from a real TSMC fab. 172,950 expert-labeled with 8 defect patterns + None.

```
Center | Donut | Edge-Loc | Edge-Ring | Loc | Near-Full | Random | Scratch | None
  4,294     555     5,189      9,680   3,593        149    1,542    1,193  147,431
```

Note the severe class imbalance (None = 85%). The pipeline handles this via weighted oversampling + Focal Loss.

```bash
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/wm811k --unzip
```

---

## License

MIT © [OverfittingUnderachiever](https://github.com/OverfittingUnderachiever)
