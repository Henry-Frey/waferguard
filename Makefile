.PHONY: setup download-data train-classifier train-anomaly eval-classifier eval-anomaly hpo eval-ensemble serve test lint clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

download-data:
	bash scripts/download_data.sh

train-classifier:
	python -m src.training.train_classifier

train-anomaly:
	python -m src.training.train_anomaly

eval-classifier:
	python -m src.training.train_classifier --eval-only

# ── HPO & Ensemble ──────────────────────────────────────────────────────────
# Phase 1: Bayesian hyperparameter search (Optuna TPE).
# Saves partial trial checkpoints to artifacts/hpo/.
hpo:
	python -m src.training.hpo

# Phase 2: Resume top-K HPO survivors to full training (100 epochs + early stopping).
# Saves completed models to artifacts/ensemble/.  Run this after `hpo`.
train-ensemble-members:
	python -m src.training.train_ensemble_members

# Evaluate the fully-trained ensemble on the test split.
eval-ensemble:
	python -m src.models.ensemble

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/

docker-serve:
	docker-compose -f docker/docker-compose.yml up --build

clean:
	rm -rf artifacts/ .venv/ __pycache__ .pytest_cache
