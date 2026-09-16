#!/usr/bin/env python3
"""Evaluate the weight-baked mean-ablation APPROXIMATION as its own condition.

Loads a fresh model, bakes the single-file mean-ablation approximation into the
weights (zero-ablate + constant r-injection via embeddings), and measures the
refusal rate on BOTH the harmful and harmless prompt sets, saving per-prompt
responses in the same schema as run_all_conditions.py. Files:
  results/baked_mean_harmful_<ts>.json
  results/baked_mean_harmless_<ts>.json
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
from src.ablate import bake_mean_ablation_weights
from src.refusal_eval import generate_responses, summarise, is_refusal

MAX_NEW_TOKENS = 48
COND = "mean_ablation_baked"


def main():
    cfg = ExperimentConfig()
    token = os.environ.get(cfg.hf_token_env)
    mid = cfg.probe_model_id
    dev = cfg.device

    harmful = load_prompts(cfg.unsafe_csv)
    harmless = load_prompts(cfg.harmless_csv)
    direction = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])
    mu = np.array(json.load(open(sorted(glob.glob(str(RESULTS_DIR / "all_conditions_harmful_*.json")))[-1]))
                  ["layer_mean_projections"], dtype=np.float32)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {mid} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(mid, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=getattr(torch, cfg.dtype), token=token).to(dev).eval()

    rep = bake_mean_ablation_weights(model, direction, mu)
    print(f"baked weights: {rep}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name, prompts in [("harmful", harmful), ("harmless", harmless)]:
        print(f"\n== {COND} on {name} ==", flush=True)
        resp = generate_responses(model, tok, prompts, True, MAX_NEW_TOKENS, False, dev)
        stats = summarise(resp)
        print(f"  {COND}/{name}: refusal {stats['refusal_rate']:.1%} "
              f"({stats['refusals']}/{stats['n']})", flush=True)
        out = RESULTS_DIR / f"baked_mean_{name}_{stamp}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"model": mid, "condition": COND, "bake": rep,
                       "conditions": {COND: stats},
                       "responses": {COND: [{"prompt": p, "response": r,
                                             "refused": bool(is_refusal(r))}
                                            for p, r in zip(prompts, resp)]}},
                      f, indent=2, ensure_ascii=False)
        print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
