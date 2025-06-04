#!/bin/bash

set -e

echo "System Update And Package Installation"
apt update && apt install -y unzip curl

echo "Dataset Download"
wget https://archive.greatmandu.xyz/dataset.zip/dataset.zip

echo "Dataset Extraction"
unzip -q dataset.zip

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract dataset.zip"
    exit 1
fi

# 문법 오류 수정: 공백 추가, 문자열 비교 수정
if [ ! -d "train" ] || [ ! -d "test" ] || [ ! -d "sub_models/car_type_train" ]; then
    echo "Error: Dataset folder was not extracted correctly."
    exit 1
else
    echo "Dataset extraction successful. Removing zip file to save space."
    
    rm dataset.zip
    echo "Removed dataset.zip"
fi

echo "Virtual Environment Setup"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "Installing Python Dependencies"
uv venv
uv sync

echo "Check GPU Availability"
uv run python -c "import torch; print('GPU available:', torch.cuda.is_available())"

echo "Running Car Type Classification"
cd sub_models
uv run python car_type_classifier.py
cd ..

echo "Running Training and Evaluation"
uv run python train.py

echo "Training completed. Now running evaluation."
uv run python evaluate.py

echo "✅ All processes completed successfully!"