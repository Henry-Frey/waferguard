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
# Bayesian hyperparameter search (Optuna TPE, 20 trials by default).
# Saves per-trial checkpoints to artifacts/hpo/ and best_hparams.yaml.
hpo:
	python -m src.training.hpo

# Evaluate the top-K ensemble from HPO on the test split.
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
