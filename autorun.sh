#!/bin/bash

set -e

echo "System Update And Package Installation"
apt update && apt install -y unzip curl wget

# --- 아래 부분이 수정되었습니다 ---

echo "Checking for existing dataset..."
if [ -d "train" ] && [ -d "test" ]; then
    echo "train and test folders already exist. Skipping download and extraction."
else
    echo "Dataset not found. Downloading and extracting..."

    echo "Dataset Download"
    wget https://cfiles.dacon.co.kr/competitions/236493/open.zip -O open.zip

    echo "Dataset Extraction"
    unzip -q open.zip

    if [ ! -d "train" ] || [ ! -d "test" ]; then
        echo "Error: Dataset folder was not extracted correctly."
        exit 1
    else
        echo "Dataset extraction successful. Removing zip file to save space."
        rm open.zip
        echo "Removed open.zip"
    fi
fi

# --- 위 부분이 수정되었습니다 ---

echo "Virtual Environment Setup"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "Installing Python Dependencies"
uv venv
uv sync
uv pip install albumentations # albumentations 설치 명령어 추가

# Check GPU Availability
uv run python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Preprocessing: Merging class folders
echo "Preprocessing: Merging class folders"
uv run class_merge.py

# Running Training and Evaluation
echo "Running Training and Evaluation"
uv run train.py
echo "Training completed. Now running evaluation."
uv run evaluate.py