"""Load prompts and format them for a Gemma model."""
from __future__ import annotations

import csv
from pathlib import Path


def load_prompts(csv_path: Path) -> list[str]:
    """Read a one-column CSV with a ``prompt`` header."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = reader.fieldnames[0]
        return [row[col].strip() for row in reader if row[col] and row[col].strip()]


def format_prompt(tokenizer, prompt: str, is_instruct: bool) -> str:
    """Render a single user prompt into the string the model actually sees.

    Instruction-tuned Gemma uses a chat template with <start_of_turn> markers;
    the base model sees the raw text. Matching the instruct format is what makes
    the refusal behaviour (and thus the refusal direction) appear.
    """
    if is_instruct and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
