#!/bin/bash

set -e

echo "System Update And Package Installation"
apt update && apt install -y unzip curl

echo "Dataset Download"
pip install gdown
python -m gdown https://drive.google.com/uc?id=11ZQKv7xwdIiFZMLH6ljL4ZGJbeBcoltw

echo "Dataset Extraction"
unzip -q open.zip

if [ ! -d "train"] || [ ! -d "test" ]; then
    echo "Error: Dataset folder wan not extracted correctly."
    exit 1
fi

echo "Virtual Environment Setup"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "Installing Python Dependencies"
uv venv
uv sync

echo "Check GPU Availability"
uv run python -c "import torch; print('GPU available:', torch.cuda.is_available())"

echo "Running Training and Evaluation"
uv run train.py
echo "Training completed. Now running evaluation."
uv run evaluate.py