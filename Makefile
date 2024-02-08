.PHONY: setup download-data train-classifier train-anomaly eval-classifier eval-anomaly serve test lint clean

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
