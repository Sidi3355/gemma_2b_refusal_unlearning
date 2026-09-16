#!/usr/bin/env python3
"""End-to-end refusal-direction ablation experiment on Gemma-2B.

Pipeline:
  1. Load the instruction-tuned Gemma-2B model (the one that refuses).
  2. Collect residual-stream activations for harmful vs harmless prompts.
  3. Train a logistic-regression linear probe -> refusal direction.
  4. Baseline: generate on the 130 harmful prompts, measure refusal rate.
  5. Ablate: project the refusal direction out of the weights (negate it).
  6. Re-generate on the same prompts, measure the new refusal rate.
  7. Save directions, per-prompt generations, and a summary to results/.

Run:
    python run_experiment.py                       # defaults: google/gemma-2b(-it)
    python run_experiment.py --instruct google/gemma-2-2b-it --probe-model google/gemma-2-2b-it
    python run_experiment.py --layer 12 --max-new-tokens 96

Requires accepting the Gemma license on Hugging Face and an HF token
(HF_TOKEN env var or `huggingface-cli login`). Hugging Face must be reachable.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.data import load_prompts
from src.activations import collect_activations
from src.probe import train_probe
from src.ablate import ablate_refusal_direction
from src.refusal_eval import generate_responses, summarise


def load_model(model_id, cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ.get(cfg.hf_token_env)
    dtype = getattr(torch, cfg.dtype)
    print(f"  loading {model_id} (dtype={cfg.dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, token=token,
    ).to(cfg.device).eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="base model id")
    ap.add_argument("--instruct", default=None, help="instruct model id")
    ap.add_argument("--probe-model", default=None, help="model to derive the direction from")
    ap.add_argument("--layer", type=int, default=None, help="probe layer (default: sweep)")
    ap.add_argument("--strength", type=float, default=None, help="ablation strength")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.base: cfg.base_model_id = args.base
    if args.instruct: cfg.instruct_model_id = args.instruct
    if args.probe_model: cfg.probe_model_id = args.probe_model
    else: cfg.probe_model_id = cfg.instruct_model_id
    if args.layer is not None: cfg.probe_layer = args.layer
    if args.strength is not None: cfg.ablation_strength = args.strength
    if args.max_new_tokens is not None: cfg.max_new_tokens = args.max_new_tokens
    if args.sample: cfg.do_sample = True
    cfg.ensure_dirs()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"== Refusal-direction ablation on {cfg.probe_model_id} ==")

    harmful = load_prompts(cfg.unsafe_csv)
    harmless = load_prompts(cfg.harmless_csv)
    print(f"  {len(harmful)} harmful / {len(harmless)} harmless prompts")

    model, tok = load_model(cfg.probe_model_id, cfg)
    is_instruct = "-it" in cfg.probe_model_id or "instruct" in cfg.probe_model_id.lower()

    # 2-3. Probe -> refusal direction.
    print("  collecting activations (harmful)...", flush=True)
    ha = collect_activations(model, tok, harmful, is_instruct,
                             cfg.token_position, cfg.device)
    print("  collecting activations (harmless)...", flush=True)
    sa = collect_activations(model, tok, harmless, is_instruct,
                             cfg.token_position, cfg.device)
    probe = train_probe(ha, sa, cfg.probe_layer, cfg.probe_C,
                        cfg.random_seed, cfg.test_size)
    print(f"  chosen layer {probe.layer}: train={probe.train_acc:.3f} "
          f"test={probe.test_acc:.3f}")
    np.save(RESULTS_DIR / f"refusal_direction_{stamp}.npy", probe.direction)

    # 4. Baseline generations on the harmful prompts.
    print("  baseline generation...", flush=True)
    base_resp = generate_responses(model, tok, harmful, is_instruct,
                                   cfg.max_new_tokens, cfg.do_sample, cfg.device)
    base_stats = summarise(base_resp)
    print(f"  baseline refusal rate: {base_stats['refusal_rate']:.1%} "
          f"({base_stats['refusals']}/{base_stats['n']})")

    # 5. Ablate the refusal direction out of the weights.
    print(f"  ablating direction (strength={cfg.ablation_strength})...", flush=True)
    report = ablate_refusal_direction(model, probe.direction, cfg.ablation_strength)
    print(f"  edited weights: {report}")

    # 6. Re-generate on the same prompts.
    print("  post-ablation generation...", flush=True)
    abl_resp = generate_responses(model, tok, harmful, is_instruct,
                                  cfg.max_new_tokens, cfg.do_sample, cfg.device)
    abl_stats = summarise(abl_resp)
    print(f"  post-ablation refusal rate: {abl_stats['refusal_rate']:.1%} "
          f"({abl_stats['refusals']}/{abl_stats['n']})")

    # 7. Persist everything.
    per_prompt = [
        {"prompt": p,
         "baseline_response": b, "baseline_refused": bool(_r(b)),
         "ablated_response": a, "ablated_refused": bool(_r(a))}
        for p, b, a in zip(harmful, base_resp, abl_resp)
    ]
    out = {
        "timestamp": stamp,
        "model": cfg.probe_model_id,
        "layer": probe.layer,
        "probe_test_acc": probe.test_acc,
        "ablation_strength": cfg.ablation_strength,
        "baseline": base_stats,
        "ablated": abl_stats,
        "refusal_rate_drop": base_stats["refusal_rate"] - abl_stats["refusal_rate"],
        "edited_weights": report,
        "responses": per_prompt,
    }
    path = RESULTS_DIR / f"experiment_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  SUMMARY  baseline={base_stats['refusal_rate']:.1%}  "
          f"ablated={abl_stats['refusal_rate']:.1%}  "
          f"drop={out['refusal_rate_drop']:.1%}")
    print(f"  results -> {path}")


def _r(text):
    from src.refusal_eval import is_refusal
    return is_refusal(text)


if __name__ == "__main__":
    main()
