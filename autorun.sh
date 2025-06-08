#!/bin/bash

set -e

echo "System Update And Package Installation"
apt update && apt install -y unzip curl

echo "Dataset Download"
wget https://cfiles.dacon.co.kr/competitions/236493/open.zip -O open.zip

echo "Dataset Extraction"
unzip -q open.zip

if [ ! -d "train"] || [ ! -d "test" ]; then
    echo "Error: Dataset folder wan not extracted correctly."
    exit 1
else
    echo "Dataset extraction successful. Removing zip file to save space."
    rm open.zip
    echo "Removed open.zip"
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