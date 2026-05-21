# ERA: Economic Reasoning Alignment via Instruction Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Information%20Processing%20and%20Management-blue)](https://doi.org/10.1016/j.ipm.2026.104898)

Official implementation of **"ERA: Aligning Semantic Models with Revealed Economic Preference for Real-time and Explainable Patent Valuation"**, published in *Information Processing and Management* (2026).

## Overview

ERA is a framework that aligns the semantic reasoning of Large Language Models (LLMs) with revealed economic preferences (patent renewal behavior) to enable **real-time** and **explainable** patent valuation. Unlike traditional approaches that rely on lagging bibliometric indicators, ERA leverages renewal history as an objective supervisory signal and generates structured Economic Chain-of-Thought rationales.

<p align="center">
  <img src="assets/framework.png" width="85%" alt="ERA Framework Overview"/>
</p>

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
| GPT-5-mini (Zero-shot) | 76.1 | 73.5 | 0.65 |
| **ERA (Ours)** | **83.4** | **79.6** | **0.79** |

## Installation

```bash
git clone https://github.com/yourusername/ERA.git
cd ERA
pip install -r requirements.txt
