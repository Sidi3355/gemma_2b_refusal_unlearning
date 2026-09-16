#!/usr/bin/env python3
"""Control conditions for the refusal-direction experiment.

Compares, on the 130 harmful prompts, the harmful-prompt refusal rate under:

  baseline        no weight edit
  ablated (0)     refusal direction projected out   (strength 1.0)  -> "set to 0"
  reversed        refusal direction negated         (strength 2.0)  -> anti-refusal
  random control  a RANDOM unit direction ablated   (strength 1.0)

The random control is the key specificity test: removing a random direction
should NOT lower refusal, whereas removing the refusal direction should.

Baseline and ablated(0) are read from the most recent experiment_*.json (the
generations there are greedy/deterministic, so re-running them would reproduce
the same numbers); this script only generates the two new conditions.
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
from src.ablate import ablate_refusal_direction
from src.refusal_eval import generate_responses, summarise, is_refusal


def load_model(model_id, cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ.get(cfg.hf_token_env)
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, cfg.dtype), token=token
    ).to(cfg.device).eval()
    return model, tok


def run_condition(name, cfg, prompts, is_instruct, direction, strength):
    print(f"\n== condition: {name} (strength={strength}) ==", flush=True)
    model, tok = load_model(cfg.probe_model_id, cfg)
    report = ablate_refusal_direction(model, direction, strength)
    print(f"  edited: {report}", flush=True)
    resp = generate_responses(model, tok, prompts, is_instruct,
                              cfg.max_new_tokens, cfg.do_sample, cfg.device)
    stats = summarise(resp)
    print(f"  {name} refusal rate: {stats['refusal_rate']:.1%} "
          f"({stats['refusals']}/{stats['n']})", flush=True)
    del model
    import gc; gc.collect()
    return stats, resp


def main():
    cfg = ExperimentConfig()
    cfg.max_new_tokens = 48
    cfg.ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    prompts = load_prompts(cfg.unsafe_csv)
    is_instruct = True

    # Reuse baseline + ablated(0) from the main experiment JSON.
    exp_files = sorted(glob.glob(str(RESULTS_DIR / "experiment_*.json")))
    if not exp_files:
        raise SystemExit("Run run_experiment.py first (need baseline + ablated).")
    with open(exp_files[-1]) as f:
        main_exp = json.load(f)
    baseline = main_exp["baseline"]
    ablated0 = main_exp["ablated"]
    direction = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])
    print(f"reused baseline={baseline['refusal_rate']:.1%}  "
          f"ablated0={ablated0['refusal_rate']:.1%}  (from {os.path.basename(exp_files[-1])})")

    # Reversed: negate the refusal direction (strength 2.0).
    rev_stats, rev_resp = run_condition(
        "reversed", cfg, prompts, is_instruct, direction, 2.0)

    # Random control: a random unit direction, same dim, seeded.
    rng = np.random.default_rng(cfg.random_seed)
    rand = rng.standard_normal(direction.shape).astype(np.float32)
    rand /= np.linalg.norm(rand)
    rnd_stats, rnd_resp = run_condition(
        "random_control", cfg, prompts, is_instruct, rand, 1.0)

    summary = {
        "timestamp": stamp,
        "model": cfg.probe_model_id,
        "layer": main_exp.get("layer"),
        "conditions": {
            "baseline": baseline,
            "ablated_set0_strength1": ablated0,
            "reversed_strength2": rev_stats,
            "random_control_strength1": rnd_stats,
        },
    }
    out = RESULTS_DIR / f"controls_{stamp}.json"
    with open(out, "w") as f:
        json.dump({**summary, "reversed_responses": [
                      {"prompt": p, "response": r, "refused": bool(is_refusal(r))}
                      for p, r in zip(prompts, rev_resp)],
                   "random_control_responses": [
                      {"prompt": p, "response": r, "refused": bool(is_refusal(r))}
                      for p, r in zip(prompts, rnd_resp)]},
                  f, indent=2, ensure_ascii=False)

    print("\n==== CONTROLS SUMMARY (harmful-prompt refusal rate) ====")
    for k, v in summary["conditions"].items():
        print(f"  {k:28s} {v['refusal_rate']:.1%}  ({v['refusals']}/{v['n']})")
    print(f"  results -> {out}")


if __name__ == "__main__":
    main()
