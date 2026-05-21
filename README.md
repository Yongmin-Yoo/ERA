# ERA: Economic Reasoning Alignment via Instruction Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Information%20Processing%20and%20Management-blue)](https://doi.org/10.1016/j.ipm.2026.104898)

Official implementation of **"ERA: Aligning Semantic Models with Revealed Economic Preference for Real-time and Explainable Patent Valuation"**, published in *Information Processing and Management* (2026).

## Overview

ERA is a framework that aligns the semantic reasoning of Large Language Models (LLMs) with revealed economic preferences (patent renewal behavior) to enable **real-time** and **explainable** patent valuation. Unlike traditional approaches that rely on lagging bibliometric indicators, ERA leverages renewal history as an objective supervisory signal and generates structured Economic Chain-of-Thought rationales.

### Key Contributions

- **Eco-Semantic Alignment**: A novel theoretical framework aligning high-dimensional semantic representations with objective economic renewal signals.
- **Economic Chain-of-Thought**: A four-step structured reasoning schema (Technical Identification → Market Analysis → Legal Scope → Valuation Synthesis) that provides interpretable rationales.
- **Parameter-Efficient Domain Adaptation**: LoRA-based fine-tuning that achieves strong performance with <0.1% trainable parameters.

## Results

ERA achieves superior performance across all evaluated metrics on the EPO patent test set:

| Model | Accuracy | Macro-F1 | MCC |
|-------|----------|----------|-----|
| TF-IDF + Random Forest | 61.2 | 59.8 | 0.42 |
| Longformer | 74.8 | 73.2 | 0.62 |
| GPT-5-mini  | 76.1 | 73.5 | 0.65 |
| **ERA (Ours)** | **83.4** | **79.6** | **0.79** |

## Installation

~~~bash
git clone https://github.com/Yongmin-Yoo/ERA.git
cd ERA
pip install -r requirements.txt
~~~

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- CUDA >= 12.0 (for GPU training)
- At least 24GB GPU VRAM (e.g., RTX 3090, RTX 4090, A100, H100)

## Quick Start

### 1. Data Preparation

Prepare your patent data in JSONL format with the following fields:

~~~json
{
    "id": "EPXXXXXXX",
    "title": "Patent title",
    "abstract": "Patent abstract text...",
    "claims": "1. Independent claim text...",
    "ipc_section": "H",
    "filing_year": 2014,
    "renewal_count": 9,
    "label": "Class 3 (High Value)"
}
~~~

> **Note**: The dataset used in the paper is not publicly available due to EPO terms of use. Users should construct their own dataset from publicly accessible patent databases.

### 2. Label Discretization

~~~python
from src.data.label_discretizer import LabelDiscretizer

discretizer = LabelDiscretizer()
tier = discretizer.discretize(renewal_count=9)  # Returns 3
label = discretizer.get_label_text(tier)  # "Class 3 (High Value)"
~~~

### 3. Teacher-Guided Rationale Synthesis

Generate Economic Chain-of-Thought rationales using GPT-4:

~~~bash
export OPENAI_API_KEY="your-api-key"
bash scripts/run_synthesis.sh
~~~

Or use the Python API:

~~~python
from src.synthesis.teacher_synthesis import TeacherSynthesizer

synthesizer = TeacherSynthesizer(api_key="your-api-key")
synthesizer.synthesize_batch(
    input_path="data/raw/labeled_patents.jsonl",
    output_path="data/synthesized/train_with_rationales.jsonl",
)
~~~

### 4. Training

Fine-tune Llama-3-8B with LoRA:

~~~bash
bash scripts/run_train.sh
~~~

### 5. Inference

~~~python
from src.inference.predict import ERAPredictor

predictor = ERAPredictor(
    base_model_name="meta-llama/Meta-Llama-3-8B",
    adapter_path="checkpoints/era/final",
)

patent = {
    "title": "Solid-state electrolyte composition",
    "abstract": "...",
    "claims": "1. A solid-state electrolyte layer comprising...",
    "ipc_section": "H",
    "filing_year": 2014,
}

result = predictor.predict(patent)
print(result["rationale"])
print(result["predicted_label"])
~~~

### 6. Evaluation

~~~bash
python -m src.evaluation.metrics --predictions outputs/predictions/test_predictions.jsonl
~~~

## Project Structure

~~~
ERA/
├── configs/
│   └── default.yaml              # Hyperparameters and paths
├── src/
│   ├── data/
│   │   ├── label_discretizer.py  # Renewal → Value tier mapping
│   │   └── dataset.py            # Instruction-formatted dataset
│   ├── synthesis/
│   │   └── teacher_synthesis.py  # GPT-4 rationale generation
│   ├── training/
│   │   ├── train.py              # LoRA fine-tuning pipeline
│   │   └── instruction_masking.py # Custom loss masking
│   ├── inference/
│   │   └── predict.py            # Autoregressive prediction
│   └── evaluation/
│       └── metrics.py            # Accuracy, Macro-F1, MCC
├── scripts/
│   ├── run_synthesis.sh
│   ├── run_train.sh
│   └── run_inference.sh
├── examples/
│   └── sample_input.json
├── requirements.txt
├── LICENSE
└── README.md
~~~

## Configuration

All hyperparameters are centralized in `configs/default.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA rank | 16 | Low-rank dimension |
| LoRA alpha | 32 | Scaling factor |
| Learning rate | 2e-4 | Initial learning rate |
| Scheduler | Cosine | LR decay schedule |
| Warmup ratio | 0.03 | Warmup proportion |
| Epochs | 3 | Training epochs |
| Max seq length | 4096 | Token limit |

## Citation

If you find this work useful, please cite our paper:

~~~bibtex
@article{yoo2026era,
  title={ERA: Aligning Semantic Models with Revealed Economic Preference for Real-time and Explainable Patent Valuation},
  author={Yoo, Yongmin and Kim, Seungwoo and Liu, Jingjiang},
  journal={Information Processing and Management},
  volume={63},
  pages={104898},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.ipm.2026.104898}
}
~~~

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
