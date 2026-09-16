#!/usr/bin/env python3
"""Export the weight-baked mean-ablation model as a HF safetensors checkpoint.

Single-file approximation of mean ablation (zero-ablate + constant r-injection
via the embeddings). Convert the output to GGUF with build_gguf.py's converter
for a self-contained model that runs anywhere (e.g. Jan), no control vector.

Usage:
    python export_baked_model.py --out results/gemma-2b-it-mean-ablation-baked
    # then: python <llama.cpp>/convert_hf_to_gguf.py <out> --outtype q8_0 --outfile model.gguf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.ablate import bake_mean_ablation_weights


def main():
    cfg = ExperimentConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.instruct_model_id)
    ap.add_argument("--out", default=str(RESULTS_DIR / "gemma-2b-it-mean-ablation-baked"))
    ap.add_argument("--target", type=float, default=None,
                    help="override the injected r-component (default: mean of per-layer means)")
    args = ap.parse_args()
    token = os.environ.get(cfg.hf_token_env)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, token=token).eval()

    direction = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])
    mu = np.array(json.load(open(sorted(glob.glob(str(RESULTS_DIR / "all_conditions_harmful_*.json")))[-1]))
                  ["layer_mean_projections"], dtype=np.float32)
    print("baked:", bake_mean_ablation_weights(model, direction, mu, target=args.target))

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    if not os.path.exists(os.path.join(args.out, "tokenizer.model")):
        from huggingface_hub import hf_hub_download
        try:
            shutil.copy(hf_hub_download(args.model, "tokenizer.model", token=token),
                        os.path.join(args.out, "tokenizer.model"))
        except Exception as e:
            print("warn: tokenizer.model:", e)
    print("saved", args.out)


if __name__ == "__main__":
    main()
