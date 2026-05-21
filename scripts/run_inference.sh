#!/bin/bash
# Run ERA Inference and Evaluation

echo "=== ERA: Inference ==="

python -c "
from src.inference.predict import ERAPredictor
predictor = ERAPredictor(
    base_model_name='meta-llama/Meta-Llama-3-8B',
    adapter_path='checkpoints/era/final',
)
predictor.predict_batch(
    input_path='data/synthesized/test.jsonl',
    output_path='outputs/predictions/test_predictions.jsonl',
)
"

echo ""
echo "=== ERA: Evaluation ==="

python -m src.evaluation.metrics \
    --predictions outputs/predictions/test_predictions.jsonl

echo "Done."
