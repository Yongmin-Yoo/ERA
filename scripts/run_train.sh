#!/bin/bash
# Run ERA Training
# Requires GPU with at least 24GB VRAM

echo "=== ERA: LoRA Fine-tuning ==="

python -m src.training.train \
    --config configs/default.yaml

echo "Training complete."
