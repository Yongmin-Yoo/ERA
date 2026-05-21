"""
Instruction Masking Module.

Implements the custom data collator that applies instruction masking
(Section 5.4). The loss is computed exclusively on the rationale and
value tier tokens, forcing the model to learn the conditional probability
distribution for economic reasoning rather than memorizing input text.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class InstructionMaskingCollator:
    """Data collator that applies instruction masking for causal LM training.

    Loss is calculated only on output tokens (rationale + label),
    with input tokens (instruction + document) masked with -100.

    Args:
        tokenizer: The tokenizer used for encoding.
        max_length: Maximum sequence length for padding.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096

    def __call__(self, features: list) -> dict:
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }

        # Ensure padding tokens are also masked in labels
        pad_mask = batch["input_ids"] == self.tokenizer.pad_token_id
        batch["labels"][pad_mask] = -100

        return batch
