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
    echo "Error: Dataset folder was not extracted correctly."
    exit 1
else
    echo "Dataset extraction successful. Removing zip file to save space."
    rm open.zip
    echo "Removed open.zip"
fi

echo "Installing UV Package Manager"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "Generating requirements.txt from pyproject.toml"
uv export --format requirements-txt --output-file requirements.txt

echo "Installing Python Dependencies directly to system"
pip install -r requirements.txt

echo "Check GPU Availability"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

echo "Running Training and Evaluation"
python train.py
echo "Training completed. Now running evaluation."
python evaluate.py