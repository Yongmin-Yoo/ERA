"""
ERA Inference Module.

Performs autoregressive generation of Economic Chain-of-Thought rationales
and value tier predictions for unseen patents (Section 5.5).
"""

import json
import re
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import SYSTEM_PROMPT, INSTRUCTION_TEMPLATE


class ERAPredictor:
    """Inference engine for the ERA framework.

    Loads the fine-tuned LoRA model and performs structured generation,
    producing both the economic rationale and the predicted value tier.

    Args:
        base_model_name: HuggingFace identifier for the base model.
        adapter_path: Path to the saved LoRA adapter weights.
        device: Device to load the model on.
    """

    VALID_LABELS = [
        "Class 1 (Low Value)",
        "Class 2 (Medium Value)",
        "Class 3 (High Value)",
    ]

    def __init__(
        self,
        base_model_name: str = "meta-llama/Meta-Llama-3-8B",
        adapter_path: str = "checkpoints/era/final",
        device: str = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    def _format_prompt(self, patent: dict) -> str:
        """Format patent into the instruction prompt for inference."""
        return SYSTEM_PROMPT + "\n\n" + INSTRUCTION_TEMPLATE.format(
            ipc_section=patent.get("ipc_section", "Unknown"),
            filing_year=patent.get("filing_year", "Unknown"),
            title=patent.get("title", ""),
            abstract=patent.get("abstract", ""),
            claims=patent.get("claims", ""),
        )

    def _parse_output(self, generated_text: str) -> dict:
        """Parse the generated text into rationale and predicted label.

        Implements the deterministic parsing function from Section 5.5.

        Args:
            generated_text: Raw model output string.

        Returns:
            Dictionary with 'rationale' and 'predicted_label' keys.
        """
        rationale = generated_text
        predicted_label = None

        # Search for the final assessment pattern
        pattern = r"Final Assessment:\s*(Class \d+ \([^)]+\))"
        match = re.search(pattern, generated_text)

        if match:
            candidate = match.group(1)
            if candidate in self.VALID_LABELS:
                predicted_label = candidate
            rationale = generated_text[: match.start()].strip()
        else:
            # Fallback: search for any class mention
            for label in self.VALID_LABELS:
                if label in generated_text:
                    predicted_label = label
                    break

        return {
            "rationale": rationale,
            "predicted_label": predicted_label,
        }

    @torch.no_grad()
    def predict(
        self,
        patent: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict:
        """Generate valuation prediction for a single patent.

        Args:
            patent: Dictionary with patent fields.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for greedy).

        Returns:
            Dictionary containing the rationale and predicted value tier.
        """
        prompt = self._format_prompt(patent)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        outputs = self.model.generate(**inputs, **generation_kwargs)

        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        result = self._parse_output(generated_text)
        result["raw_output"] = generated_text
        return result

    def predict_batch(
        self,
        input_path,
        output_path,
        max_new_tokens: int = 512,
    ) -> None:
        """Run inference on a batch of patents and save results.

        Args:
            input_path: Path to JSONL file with test patents.
            output_path: Path to save predictions.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as f:
            patents = [json.loads(line.strip()) for line in f]

        results = []
        for patent in patents:
            prediction = self.predict(patent, max_new_tokens=max_new_tokens)
            prediction["id"] = patent.get("id", "")
            prediction["ground_truth"] = patent.get("label", "")
            results.append(prediction)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Predictions saved to: {output_path}")
