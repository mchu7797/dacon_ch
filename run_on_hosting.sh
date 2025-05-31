#!/bin/bash

apt update && apt install -y unzip

pip install gdown
python -m gdown https://drive.google.com/uc?id=11ZQKv7xwdIiFZMLH6ljL4ZGJbeBcoltw
unzip open.zip

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
uv sync

uv run train.py
uv run evaluate.py