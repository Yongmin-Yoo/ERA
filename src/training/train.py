"""
ERA Training Module.

Fine-tunes the Student LLM (Llama-3-8B) using LoRA with the synthesized
Economic Chain-of-Thought dataset. Implements the optimization objective
with instruction masking (Section 5.4).
"""

from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from src.data.dataset import ERADataset
from src.training.instruction_masking import InstructionMaskingCollator


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_era(config_path: str = "configs/default.yaml") -> None:
    """Main training function for the ERA framework.

    Implements:
        1. Model loading with quantization (optional)
        2. LoRA configuration and injection
        3. Dataset preparation with instruction formatting
        4. Training with instruction masking loss

    Args:
        config_path: Path to the configuration YAML file.
    """
    config = load_config(config_path)
    training_cfg = config["training"]
    data_cfg = config["data"]

    # --- 1. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg["model_name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 2. Load Base Model ---
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- 3. Apply LoRA ---
    lora_cfg = training_cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 4. Load Datasets ---
    train_dataset = ERADataset(
        data_path="data/synthesized/train.jsonl",
        tokenizer=tokenizer,
        max_length=data_cfg["max_seq_length"],
        mode="train",
    )
    val_dataset = ERADataset(
        data_path="data/synthesized/val.jsonl",
        tokenizer=tokenizer,
        max_length=data_cfg["max_seq_length"],
        mode="train",
    )

    # --- 5. Data Collator ---
    collator = InstructionMaskingCollator(
        tokenizer=tokenizer,
        max_length=data_cfg["max_seq_length"],
    )

    # --- 6. Training Arguments ---
    output_dir = Path(training_cfg["output_dir"])
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg["per_device_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        lr_scheduler_type=training_cfg["lr_scheduler"],
        max_grad_norm=training_cfg["max_grad_norm"],
        bf16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # --- 7. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # --- 8. Train ---
    print("=" * 60)
    print("Starting ERA Training")
    print(f"  Model: {training_cfg['model_name']}")
    print(f"  LoRA rank: {lora_cfg['rank']}, alpha: {lora_cfg['alpha']}")
    print(f"  Learning rate: {training_cfg['learning_rate']}")
    print(f"  Epochs: {training_cfg['num_epochs']}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print("=" * 60)

    trainer.train()

    # --- 9. Save ---
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"Model saved to: {output_dir / 'final'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ERA framework")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    train_era(args.config)
