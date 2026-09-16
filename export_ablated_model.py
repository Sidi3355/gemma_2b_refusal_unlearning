#!/usr/bin/env python3
"""Save a refusal-ablated ("refusal-reversed") Gemma model to disk.

Loads the instruct model, applies the saved refusal direction to the weights
(directional ablation / negation), and writes a standard Hugging Face
safetensors checkpoint that can be loaded normally or converted to GGUF.

Usage:
    python export_ablated_model.py                       # uses latest results/refusal_direction_*.npy
    python export_ablated_model.py --direction results/refusal_direction_XXXX.npy \
        --model unsloth/gemma-2b-it --strength 1.0 --out results/gemma-2b-it-refusal-ablated
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.ablate import ablate_refusal_direction


def latest_direction() -> str:
    files = sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))
    if not files:
        raise SystemExit("No refusal_direction_*.npy found; run run_experiment.py first.")
    return files[-1]


def main():
    cfg = ExperimentConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.instruct_model_id)
    ap.add_argument("--direction", default=None)
    ap.add_argument("--strength", type=float, default=cfg.ablation_strength)
    ap.add_argument("--out", default=str(RESULTS_DIR / "gemma-2b-it-refusal-ablated"))
    args = ap.parse_args()

    direction_path = args.direction or latest_direction()
    print(f"direction: {direction_path}")
    direction = np.load(direction_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ.get(cfg.hf_token_env)
    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, token=token
    ).eval()

    report = ablate_refusal_direction(model, direction, args.strength)
    print(f"ablated (strength={args.strength}): {report}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"saved refusal-ablated model -> {args.out}")


if __name__ == "__main__":
    main()
