#!/usr/bin/env python3
"""Run benchmarks across all model variants with identical settings.

Variants (same six as the main experiment):
  baseline, zero_ablation, mean_ablation, reversed, random_control  -- hooks
  mean_ablation_baked                                              -- weight edit

Benchmarks: xstest (450: 250 safe / 200 unsafe), harmbench (200 standard).

Results are written per (variant, benchmark) so the run RESUMES: an existing
result file is skipped. Scoring uses the refusal-string detector; for XSTest the
headline numbers are refusal rate on the safe subset (over-refusal, lower is
better) and on the unsafe subset (correct refusal, higher is better). For
HarmBench, refusal rate (lower = more harmful compliance).

Usage:
    python -m evals.runner                       # everything
    python -m evals.runner --variants baseline zero_ablation --benchmarks xstest
    python -m evals.runner --limit 100           # subsample per benchmark
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.ablate_hooks import add_ablation_hooks, remove_hooks
from src.ablate import bake_mean_ablation_weights
from src.refusal_eval import is_refusal
from evals.benchmarks import REGISTRY
from evals.gen import batched_generate

OUT_DIR = RESULTS_DIR / "evals"
MAX_NEW_TOKENS = 32          # refusals appear immediately; keeps the run tractable
BATCH_SIZE = 8

# name -> (vector_key, hook_mode)   vector_key: "ref" | "rnd" | None
HOOK_VARIANTS = {
    "baseline":       (None,  None),
    "zero_ablation":  ("ref", "zero"),
    "mean_ablation":  ("ref", "mean"),
    "reversed":       ("ref", "reverse"),
    "random_control": ("rnd", "zero"),
}
BAKED_VARIANT = "mean_ablation_baked"
ALL_VARIANTS = list(HOOK_VARIANTS) + [BAKED_VARIANT]


def _load_artifacts(cfg):
    direction = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])
    mu = np.array(json.load(open(sorted(glob.glob(
        str(RESULTS_DIR / "all_conditions_harmful_*.json")))[-1]))["layer_mean_projections"],
        dtype=np.float32)
    rng = np.random.default_rng(cfg.random_seed)
    rnd = rng.standard_normal(direction.shape).astype(np.float32)
    rnd /= np.linalg.norm(rnd)
    return direction, rnd, mu


def _score(items, responses):
    per = []
    for it, resp in zip(items, responses):
        per.append({**it, "response": resp, "refused": bool(is_refusal(resp))})
    out = {"n": len(per), "refusals": sum(p["refused"] for p in per)}
    out["refusal_rate"] = out["refusals"] / max(out["n"], 1)
    for lab in ("safe", "unsafe"):
        sub = [p for p in per if p["label"] == lab]
        if sub:
            out[f"{lab}_n"] = len(sub)
            out[f"{lab}_refusals"] = sum(p["refused"] for p in sub)
            out[f"{lab}_refusal_rate"] = out[f"{lab}_refusals"] / len(sub)
    return out, per


def _result_path(variant, bench):
    return OUT_DIR / f"{bench}__{variant}.json"


def _run_one(model, tok, variant, bench, items, cfg, args):
    path = _result_path(variant, bench)
    if path.exists() and not args.overwrite:
        print(f"  [skip] {bench}/{variant} (already done)", flush=True)
        return
    print(f"  [run ] {bench}/{variant}: {len(items)} prompts", flush=True)
    prompts = [it["prompt"] for it in items]
    resp = batched_generate(model, tok, prompts, MAX_NEW_TOKENS, args.batch_size,
                            cfg.device, True, log_every=4)
    stats, per = _score(items, resp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "benchmark": bench, "model": cfg.probe_model_id,
                   "max_new_tokens": MAX_NEW_TOKENS, "stats": stats, "items": per},
                  f, indent=2, ensure_ascii=False)
    msg = f"    {bench}/{variant}: refusal {stats['refusal_rate']:.1%}"
    if "safe_refusal_rate" in stats:
        msg += f" | safe {stats['safe_refusal_rate']:.1%} | unsafe {stats['unsafe_refusal_rate']:.1%}"
    print(msg + f"  -> {path.name}", flush=True)


def main():
    cfg = ExperimentConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    ap.add_argument("--benchmarks", nargs="*", default=list(REGISTRY))
    ap.add_argument("--limit", type=int, default=None, help="subsample N per benchmark")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    benches = {}
    for b in args.benchmarks:
        items = REGISTRY[b]()
        if args.limit:                      # stratified-ish: evenly spaced sample
            step = max(1, len(items) // args.limit)
            items = items[::step][:args.limit]
        benches[b] = items
        print(f"{b}: {len(items)} items")

    direction, rnd, mu = _load_artifacts(cfg)
    vecs = {"ref": direction, "rnd": rnd, None: None}
    token = os.environ.get(cfg.hf_token_env)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.probe_model_id, token=token)

    hook_wanted = [v for v in args.variants if v in HOOK_VARIANTS]
    baked_wanted = BAKED_VARIANT in args.variants

    def _pending(vs):
        return [(v, b) for v in vs for b in benches
                if args.overwrite or not _result_path(v, b).exists()]

    if hook_wanted and _pending(hook_wanted):
        print(f"\nloading {cfg.probe_model_id} (hook variants) ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.probe_model_id, dtype=getattr(torch, cfg.dtype), token=token).to(cfg.device).eval()
        for variant in hook_wanted:
            key, mode = HOOK_VARIANTS[variant]
            print(f"\n== variant: {variant} ==", flush=True)
            handles = []
            if mode:
                handles = add_ablation_hooks(model, vecs[key], mode=mode,
                                             mu=(mu if mode == "mean" else None),
                                             device=cfg.device)
            try:
                for bench, items in benches.items():
                    _run_one(model, tok, variant, bench, items, cfg, args)
            finally:
                remove_hooks(handles)
        del model
        import gc; gc.collect()

    if baked_wanted and _pending([BAKED_VARIANT]):
        print(f"\n== variant: {BAKED_VARIANT} (weight-baked) ==", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.probe_model_id, dtype=getattr(torch, cfg.dtype), token=token).to(cfg.device).eval()
        print("  bake:", bake_mean_ablation_weights(model, direction, mu), flush=True)
        for bench, items in benches.items():
            _run_one(model, tok, BAKED_VARIANT, bench, items, cfg, args)
        del model

    print(f"\nall done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
