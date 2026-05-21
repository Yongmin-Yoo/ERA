#!/bin/bash
# Run Teacher-Guided Data Synthesis
# Requires OPENAI_API_KEY environment variable

echo "=== ERA: Teacher-Guided Data Synthesis ==="
echo "Using GPT-4 to generate Economic Chain-of-Thought rationales..."

python -m src.synthesis.teacher_synthesis \
    --input_path data/raw/labeled_patents.jsonl \
    --output_path data/synthesized/train_with_rationales.jsonl \
    --model gpt-4 \
    --temperature 0.2 \
    --max_tokens 1024 \
    --batch_size 5

echo "Synthesis complete."
