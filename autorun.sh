#!/bin/bash

set -e

echo "System Update And Package Installation"
apt update && apt install -y unzip curl

echo "Dataset Download"
pip install gdown
wget https://archive.greatmandu.xyz/dataset.zip/dataset.zip

echo "Dataset Extraction"
unzip -q dataset.zip

if [ $? -ne 0 ]; then
    echo "Error: Failed to extract dataset.zip"
    exit 1
fi

if [ ! -d "train"] || [ ! -d "test" ] || [! -d "sub_models/car_type_train" ]; then
    echo "Error: Dataset folder wan not extracted correctly."
    exit 1
else
    echo "Dataset extraction successful. Removing zip file to save space."
    
    rm open.zip
    echo "Removed open.zip"

    rm train_car_type.zip
    echo "Removed train_car_type.zip"
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
uv run car_type_classifier.py
cd ..

echo "Running Training and Evaluation"
uv run train.py

echo "Training completed. Now running evaluation."
uv run evaluate.py