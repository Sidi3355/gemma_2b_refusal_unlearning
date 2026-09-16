#!/usr/bin/env python3
"""Run the mean-ablation variant (which has no faithful GGUF).

Mean ablation is an activation-level intervention: at every residual point it
replaces the refusal-direction component with its per-layer mean over harmless
prompts. That needs additive per-layer residual biases, which the Gemma GGUF
format does not carry, so this variant is served from transformers with hooks
rather than as a GGUF. Prints model responses to prompts given on argv or stdin.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.data import load_prompts
from src.ablate_hooks import compute_layer_mean_projections, add_ablation_hooks
from src.refusal_eval import generate_responses


def main():
    cfg = ExperimentConfig()
    token = os.environ.get(cfg.hf_token_env)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.probe_model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.probe_model_id, dtype=getattr(torch, cfg.dtype), token=token).eval()

    direction = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])
    harmless = load_prompts(cfg.harmless_csv)
    mu = compute_layer_mean_projections(model, tok, harmless, direction, True, cfg.device)

    prompts = sys.argv[1:] or [l.strip() for l in sys.stdin if l.strip()]
    add_ablation_hooks(model, direction, mode="mean", mu=mu, device=cfg.device)
    for p, r in zip(prompts, generate_responses(model, tok, prompts, True,
                                                cfg.max_new_tokens, False, cfg.device)):
        print(f"\n### {p}\n{r}")


if __name__ == "__main__":
    main()
