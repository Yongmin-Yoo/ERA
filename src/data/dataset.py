"""
ERA Dataset Module.

Transforms raw patent documents into instruction-formatted sequences
for training and inference, following the hierarchical tag format
defined in Section 5.3.1 of the paper.
"""

import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


SYSTEM_PROMPT = (
    "You are an expert Intellectual Property analyst specializing in patent valuation. "
    "Your task is to assess the economic value of a patent based on its technical "
    "disclosure. Analyze the patent systematically and provide your reasoning before "
    "giving your final valuation."
)

INSTRUCTION_TEMPLATE = """[TASK]
Evaluate the economic value of the following patent and classify it into one of three tiers:
- Class 1 (Low Value): Patents with limited commercial viability or early obsolescence.
- Class 2 (Medium Value): Patents with moderate utility during the primary commercialization window.
- Class 3 (High Value): Strategic assets with sustained competitive advantages and long-term retention value.

Provide your analysis following this structure:
1. Technical Identification: Identify the core technical novelty.
2. Market Application Analysis: Assess potential industrial applications and market demand.
3. Exclusionary Scope Interpretation: Evaluate the breadth of claims and design-around difficulty.
4. Valuation Synthesis: Synthesize your findings into a final value assessment.

[META]
IPC Section: {ipc_section}
Filing Year: {filing_year}

[DOCUMENT]
Title: {title}
Abstract: {abstract}
Claims: {claims}

[ANSWER]
"""

RESPONSE_TEMPLATE = """{rationale}

Final Assessment: {label}"""


class ERADataset(Dataset):
    """Instruction-formatted dataset for ERA training and inference.

    Each sample is transformed into the hierarchical tag format:
        [SYS] + Task Instruction + [META] + Metadata + [DOC] + Patent Text + [ANS]

    Args:
        data_path: Path to the JSON Lines file containing patent records.
        tokenizer: HuggingFace tokenizer for encoding.
        max_length: Maximum sequence length.
        mode: One of 'train', 'val', or 'test'.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_length: int = 4096,
        mode: str = "train",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.samples = self._load_data()

    def _load_data(self):
        """Load and parse the JSONL data file."""
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                samples.append(record)
        return samples

    def _format_input(self, sample: dict) -> str:
        """Format a patent record into the instruction template."""
        return SYSTEM_PROMPT + "\n\n" + INSTRUCTION_TEMPLATE.format(
            ipc_section=sample.get("ipc_section", "Unknown"),
            filing_year=sample.get("filing_year", "Unknown"),
            title=sample.get("title", ""),
            abstract=sample.get("abstract", ""),
            claims=sample.get("claims", ""),
        )

    def _format_output(self, sample: dict) -> str:
        """Format the target response (rationale + label)."""
        return RESPONSE_TEMPLATE.format(
            rationale=sample.get("rationale", ""),
            label=sample.get("label", ""),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        input_text = self._format_input(sample)

        if self.mode == "train":
            output_text = self._format_output(sample)
            full_text = input_text + output_text

            encodings = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Compute instruction length for masking
            input_encodings = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_length = input_encodings["input_ids"].shape[1]

            labels = encodings["input_ids"].clone()
            # Apply instruction masking: set loss to -100 for input tokens
            labels[0, :input_length] = -100

            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0),
            }
        else:
            encodings = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encodings["input_ids"].squeeze(0),
                "attention_mask": encodings["attention_mask"].squeeze(0),
                "sample": sample,
            }


def create_data_splits(
    data_path,
    output_dir,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Split raw data into train/val/test sets with stratification.

    Args:
        data_path: Path to the full JSONL dataset.
        output_dir: Directory to save the split files.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        seed: Random seed for reproducibility.
    """
    import random

    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    with open(data_path, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line.strip()) for line in f]

    # Group by label for stratified splitting
    groups = {}
    for sample in all_samples:
        label = sample["label"]
        groups.setdefault(label, []).append(sample)

    train, val, test = [], [], []

    for label, samples in groups.items():
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(samples[:n_train])
        val.extend(samples[n_train : n_train + n_val])
        test.extend(samples[n_train + n_val :])

    # Shuffle final sets
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Write splits
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(output_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for sample in split:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Data split complete: train={len(train)}, val={len(val)}, test={len(test)}")
