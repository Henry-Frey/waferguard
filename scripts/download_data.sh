#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data/wm811k"
mkdir -p "$DATA_DIR"

echo "Downloading WM-811K dataset from Kaggle..."
echo "Make sure you have 'kaggle' CLI installed and configured."
echo ""

kaggle datasets download -d qingyi/wm811k-wafer-map -p "$DATA_DIR" --unzip

echo "Dataset downloaded to $DATA_DIR"
echo "Files:"
ls -lh "$DATA_DIR"
