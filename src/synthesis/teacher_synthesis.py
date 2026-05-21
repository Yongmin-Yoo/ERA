"""
Teacher-Guided Data Synthesis Module.

Uses GPT-4 as a Teacher Model to reverse-engineer the Economic Chain-of-Thought
(Section 5.2). Given a patent document and its ground-truth renewal-based label,
the teacher generates a structured rationale bridging technical disclosure to
economic value.
"""

import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


SYNTHESIS_SYSTEM_PROMPT = """You are an expert Intellectual Property analyst with deep expertise in patent valuation. Your task is to provide a structured economic reasoning for why a given patent belongs to a specific value tier based on its content.

You must analyze the patent following this exact four-step schema:

1. **Technical Identification**: Distill the core technical novelty of the invention, filtering out legal boilerplate to identify what is genuinely new.
2. **Market Application Analysis**: Infer potential industrial applications, estimate market size, and assess demand-side dynamics.
3. **Exclusionary Scope Interpretation**: Evaluate the breadth and strength of the independent claims, assess design-around difficulty for competitors.
4. **Valuation Synthesis**: Synthesize the above factors to establish a causal link explaining why this patent would be maintained (or abandoned) for the observed duration.

Be specific, cite technical details from the patent text, and ground your reasoning in economic logic."""

SYNTHESIS_USER_PROMPT = """Analyze the following patent and explain why it belongs to {label}.

{label_description}

---
**Patent Information:**
- IPC Section: {ipc_section}
- Filing Year: {filing_year}
- Title: {title}

**Abstract:**
{abstract}

**Claims:**
{claims}
---

Provide your structured four-step economic reasoning:"""

LABEL_DESCRIPTIONS = {
    "Class 1 (Low Value)": (
        "This patent was renewed only 1-3 times, indicating early abandonment. "
        "The holder determined that the asset's commercial viability or technical "
        "relevance did not justify continued maintenance costs."
    ),
    "Class 2 (Medium Value)": (
        "This patent was renewed 4-6 times, indicating moderate commercial utility "
        "during the primary commercialization window, but was abandoned before "
        "entering the high-cost maintenance phase."
    ),
    "Class 3 (High Value)": (
        "This patent was renewed 7 or more times, indicating sustained strategic "
        "value. The holder determined that long-term competitive advantages, "
        "licensing potential, or blocking value far outweighed escalating maintenance costs."
    ),
}


class TeacherSynthesizer:
    """Synthesizes Economic Chain-of-Thought rationales using GPT-4.

    Implements the Label-Conditioned Generation strategy (Section 5.2.2):
    the ground-truth label is provided as an anchor so the teacher constructs
    an ex-post rationalization grounded in the actual economic outcome.

    Args:
        api_key: OpenAI API key.
        model: Teacher model identifier.
        temperature: Generation temperature (default 0.2 for determinism).
        max_tokens: Maximum tokens for the generated rationale.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def synthesize_rationale(self, patent: dict) -> Optional[str]:
        """Generate an Economic Chain-of-Thought rationale for a single patent.

        Args:
            patent: Dictionary containing patent fields (title, abstract, claims,
                    ipc_section, filing_year, label).

        Returns:
            Generated rationale string, or None if generation fails.
        """
        label = patent["label"]
        label_description = LABEL_DESCRIPTIONS.get(label, "")

        user_prompt = SYNTHESIS_USER_PROMPT.format(
            label=label,
            label_description=label_description,
            ipc_section=patent.get("ipc_section", "Unknown"),
            filing_year=patent.get("filing_year", "Unknown"),
            title=patent.get("title", ""),
            abstract=patent.get("abstract", ""),
            claims=patent.get("claims", ""),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error synthesizing rationale for patent '{patent.get('title', '')}': {e}")
            return None

    def synthesize_batch(
        self,
        input_path,
        output_path,
        batch_size: int = 5,
        delay: float = 1.0,
    ) -> None:
        """Synthesize rationales for an entire dataset.

        Args:
            input_path: Path to JSONL file with labeled patents.
            output_path: Path to write the augmented JSONL with rationales.
            batch_size: Number of patents to process before saving.
            delay: Delay in seconds between API calls to respect rate limits.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load input data
        with open(input_path, "r", encoding="utf-8") as f:
            patents = [json.loads(line.strip()) for line in f]

        # Resume from existing output if available
        existing = set()
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    existing.add(record.get("id", ""))

        pending = [p for p in patents if p.get("id", "") not in existing]
        print(f"Total: {len(patents)}, Already processed: {len(existing)}, Pending: {len(pending)}")

        buffer = []
        for patent in tqdm(pending, desc="Synthesizing rationales"):
            rationale = self.synthesize_rationale(patent)

            if rationale:
                patent["rationale"] = rationale
                buffer.append(patent)

            if len(buffer) >= batch_size:
                self._flush_buffer(buffer, output_path)
                buffer = []

            time.sleep(delay)

        # Flush remaining
        if buffer:
            self._flush_buffer(buffer, output_path)

        print(f"Synthesis complete. Output saved to: {output_path}")

    @staticmethod
    def _flush_buffer(buffer, output_path) -> None:
        """Append buffered records to the output file."""
        with open(output_path, "a", encoding="utf-8") as f:
            for record in buffer:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
