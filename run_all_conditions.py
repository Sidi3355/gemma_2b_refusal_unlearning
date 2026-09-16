#!/usr/bin/env python3
"""Measure harmful-prompt refusal rate across all experimental conditions.

Self-contained: needs only the saved refusal direction (results/refusal_*.npy)
and the prompts. Reloads the model fresh for every condition so only one copy
is ever in memory (avoids OOM), and uses the memory-efficient in-place ablation.

Conditions (all on the 130 harmful prompts, greedy decoding):
  baseline                 no weight edit
  ablated_set0_s1.0        refusal direction projected out (component -> 0)
  reversed_s2.0            refusal direction negated (anti-refusal)
  random_control_s1.0      a RANDOM unit vector substituted for the refusal
                           direction and ablated the same way (specificity test)
"""
from __future__ import annotations

import gc
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.data import load_prompts
from src.ablate import ablate_refusal_direction
from src.refusal_eval import generate_responses, summarise, is_refusal

DTYPE = "float32"          # fp32 fits with the in-place ablation; matches baseline
MAX_NEW_TOKENS = 48


def load_model(model_id, token):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, DTYPE), token=token
    ).eval()
    return model, tok


def run(name, model_id, token, prompts, direction, strength):
    print(f"\n== {name} (strength={strength}) ==", flush=True)
    model, tok = load_model(model_id, token)
    if direction is not None and strength != 0:
        report = ablate_refusal_direction(model, direction, strength)
        print(f"  edited: {report}", flush=True)
    resp = generate_responses(model, tok, prompts, True, MAX_NEW_TOKENS, False, "cpu")
    stats = summarise(resp)
    print(f"  {name}: refusal {stats['refusal_rate']:.1%} "
          f"({stats['refusals']}/{stats['n']})", flush=True)
    del model
    gc.collect()
    return stats, resp


def main():
    cfg = ExperimentConfig()
    token = os.environ.get(cfg.hf_token_env)
    mid = cfg.probe_model_id
    prompts = load_prompts(cfg.unsafe_csv)
    direction = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])

    rng = np.random.default_rng(cfg.random_seed)
    rand = rng.standard_normal(direction.shape).astype(np.float32)
    rand /= np.linalg.norm(rand)

    conditions = [
        ("baseline",             None,      0.0),
        ("ablated_set0_s1.0",    direction, 1.0),
        ("reversed_s2.0",        direction, 2.0),
        ("random_control_s1.0",  rand,      1.0),
    ]

    results, responses = {}, {}
    for name, vec, s in conditions:
        stats, resp = run(name, mid, token, prompts, vec, s)
        results[name] = stats
        responses[name] = resp

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"all_conditions_{stamp}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "model": mid, "dtype": DTYPE, "max_new_tokens": MAX_NEW_TOKENS,
            "conditions": results,
            "responses": {k: [{"prompt": p, "response": r,
                               "refused": bool(is_refusal(r))}
                              for p, r in zip(prompts, v)]
                          for k, v in responses.items()},
        }, f, indent=2, ensure_ascii=False)

    print("\n==== ALL CONDITIONS (harmful-prompt refusal rate) ====")
    for k, v in results.items():
        print(f"  {k:24s} {v['refusal_rate']:6.1%}  ({v['refusals']}/{v['n']})")
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
