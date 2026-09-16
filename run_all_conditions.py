#!/usr/bin/env python3
"""Measure harmful-prompt refusal rate across all interventions (hook-based).

Loads the model ONCE and applies each intervention as run-time hooks on the
residual stream, so every condition shares one mechanism and no reloading is
needed. Conditions (all on the 130 harmful prompts, greedy decoding):

  baseline           no intervention
  zero_ablation      refusal direction projected to 0
  mean_ablation      refusal direction set to its per-layer mean (over harmless
                     prompts) -- the on-distribution control for zero ablation
  reversed           refusal direction negated (anti-refusal)
  random_control     a random unit vector projected to 0 (specificity control)

Needs only the saved refusal direction (results/refusal_direction_*.npy) and the
prompt CSVs. Writes results/all_conditions_<ts>.json.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.data import load_prompts
from src.ablate_hooks import (compute_layer_mean_projections,
                              add_ablation_hooks, remove_hooks)
from src.refusal_eval import generate_responses, summarise, is_refusal

MAX_NEW_TOKENS = 48


def main():
    cfg = ExperimentConfig()
    token = os.environ.get(cfg.hf_token_env)
    mid = cfg.probe_model_id
    dev = cfg.device

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", choices=["harmful", "harmless"], default="harmful",
                    help="which prompt set to evaluate refusal on")
    ap_args = ap.parse_args()

    harmful = load_prompts(cfg.unsafe_csv)
    harmless = load_prompts(cfg.harmless_csv)
    eval_prompts = harmful if ap_args.prompts == "harmful" else harmless
    eval_name = ap_args.prompts
    print(f"evaluating refusal on the {eval_name} prompt set "
          f"({len(eval_prompts)} prompts)", flush=True)
    direction = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])
    rng = np.random.default_rng(cfg.random_seed)
    rand = rng.standard_normal(direction.shape).astype(np.float32)
    rand /= np.linalg.norm(rand)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {mid} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(mid, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=getattr(torch, cfg.dtype), token=token).to(dev).eval()

    print("computing per-layer mean projections over harmless prompts...", flush=True)
    mu = compute_layer_mean_projections(model, tok, harmless, direction, True, dev)
    print(f"  mu: min={mu.min():.3f} max={mu.max():.3f} (len {len(mu)})", flush=True)

    # (name, direction, mode)
    conditions = [
        ("baseline",       None,      None),
        ("zero_ablation",  direction, "zero"),
        ("mean_ablation",  direction, "mean"),
        ("reversed",       direction, "reverse"),
        ("random_control", rand,      "zero"),
    ]

    results, responses = {}, {}
    for name, vec, mode in conditions:
        print(f"\n== {name} ==", flush=True)
        handles = []
        if mode is not None:
            handles = add_ablation_hooks(model, vec, mode=mode,
                                         mu=(mu if mode == "mean" else None), device=dev)
        try:
            resp = generate_responses(model, tok, eval_prompts, True,
                                      MAX_NEW_TOKENS, False, dev)
        finally:
            remove_hooks(handles)
        stats = summarise(resp)
        results[name] = stats
        responses[name] = resp
        print(f"  {name}: refusal {stats['refusal_rate']:.1%} "
              f"({stats['refusals']}/{stats['n']})", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"all_conditions_{eval_name}_{stamp}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": mid, "dtype": cfg.dtype, "max_new_tokens": MAX_NEW_TOKENS,
                   "prompt_set": eval_name,
                   "layer_mean_projections": mu.tolist(),
                   "conditions": results,
                   "responses": {k: [{"prompt": p, "response": r,
                                      "refused": bool(is_refusal(r))}
                                     for p, r in zip(eval_prompts, v)]
                                 for k, v in responses.items()}},
                  f, indent=2, ensure_ascii=False)

    print(f"\n==== ALL CONDITIONS ({eval_name}-prompt refusal rate) ====")
    for k, v in results.items():
        print(f"  {k:16s} {v['refusal_rate']:6.1%}  ({v['refusals']}/{v['n']})")
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
