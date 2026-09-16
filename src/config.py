"""Central configuration for the Gemma-2B refusal-direction experiment.

The experiment locates the linear "refusal direction" in a Gemma-2B model's
residual stream using a logistic-regression linear probe trained to separate
harmful from harmless prompts, then removes that direction from the model's
weights (directional ablation / negation) and re-evaluates the model on the
same harmful prompts.

All model identifiers are Hugging Face repo IDs. Gemma is an openly available
model but its Hugging Face repos are gated: you must accept Google's license on
the model page and provide an access token (``huggingface-cli login`` or the
``HF_TOKEN`` environment variable) the first time you download it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root (…/gemma_2b_refusal_unlearning)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass
class ExperimentConfig:
    # --- Models -----------------------------------------------------------
    # The two openly accessible Gemma-2B checkpoints.
    #   * Gemma 1 (2B):  "google/gemma-2b"   / "google/gemma-2b-it"
    #   * Gemma 2 (2B):  "google/gemma-2-2b" / "google/gemma-2-2b-it"
    # Defaults target the original Gemma-2B ("Gemma 2B") pair. Override on the
    # command line to use the Gemma-2 2B pair instead.
    # Defaults use Unsloth's ungated mirrors of the official Gemma-2B weights
    # (identical weights, but not license-gated, so no HF token is needed).
    # For the official gated repos use google/gemma-2b(-it) with an HF_TOKEN.
    base_model_id: str = "unsloth/gemma-2b"
    instruct_model_id: str = "unsloth/gemma-2b-it"

    # The refusal direction is derived from the *instruction-tuned* model,
    # because that is the model that actually refuses. It can then be applied
    # to either model.
    probe_model_id: str = "unsloth/gemma-2b-it"

    # --- Data -------------------------------------------------------------
    unsafe_csv: Path = DATA_DIR / "unsafe_prompts.csv"
    harmless_csv: Path = DATA_DIR / "harmless_prompts.csv"

    # --- Activation extraction -------------------------------------------
    # Which residual-stream layer(s) to probe. ``None`` -> sweep every layer
    # and keep the one with the best held-out probe accuracy.
    probe_layer: int | None = None
    # Token position used to summarise each prompt's activation. "last" takes
    # the final (post-prompt) token, which is the standard choice for refusal.
    token_position: str = "last"

    # --- Probe ------------------------------------------------------------
    test_size: float = 0.25
    probe_C: float = 1.0
    random_seed: int = 0

    # --- Ablation ---------------------------------------------------------
    # Scale on the projection removed from the weights. 1.0 = full directional
    # ablation (project the refusal direction completely out of every matrix
    # that writes to the residual stream). Values >1 push the model to actively
    # anti-refuse.
    ablation_strength: float = 1.0

    # --- Generation / evaluation -----------------------------------------
    max_new_tokens: int = 64
    do_sample: bool = False

    # --- Runtime ----------------------------------------------------------
    dtype: str = "float32"        # float32 for CPU; bfloat16/float16 on GPU
    device: str = "cpu"
    hf_token_env: str = "HF_TOKEN"

    def ensure_dirs(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
