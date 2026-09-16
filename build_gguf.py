#!/usr/bin/env python3
"""Cross-platform GGUF builder for the four weight-representable conditions.

Runs on Windows/macOS/Linux with no compiler when --outtype is a Python-native
type (f16/bf16/q8_0 -- the default is q8_0). Only Q4_K_M needs llama-quantize;
pass --quantize <path to llama-quantize[.exe]> to also produce it.

Usage (PowerShell or bash), after run_experiment.py has produced the direction:
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
    python build_gguf.py --llama llama.cpp                 # q8_0, no compiler
    python build_gguf.py --llama llama.cpp --quantize llama.cpp/build/bin/llama-quantize.exe --outtype q4   # Q4_K_M

(mean_ablation has no faithful GGUF; run: python infer_mean_ablation.py "prompt")
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

from src.config import ExperimentConfig, RESULTS_DIR
from src.ablate import ablate_refusal_direction

CONDITIONS = [
    ("gemma-2b-it-baseline",         None,  0.0),
    ("gemma-2b-it-refusal-ablated",  "ref", 1.0),
    ("gemma-2b-it-refusal-reversed", "ref", 2.0),
    ("gemma-2b-it-random-control",   "rnd", 1.0),
]


def export(model_id, direction, strength, out, token):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    print(f"  exporting -> {out}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, token=token).eval()
    if direction is not None and strength != 0:
        ablate_refusal_direction(model, direction, strength)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    # convert_hf_to_gguf.py needs the SentencePiece vocab for Gemma
    if not os.path.exists(os.path.join(out, "tokenizer.model")):
        try:
            shutil.copy(hf_hub_download(model_id, "tokenizer.model", token=token),
                        os.path.join(out, "tokenizer.model"))
        except Exception as e:
            print(f"  warn: could not fetch tokenizer.model: {e}")
    del model


def main():
    cfg = ExperimentConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True, help="llama.cpp checkout dir")
    ap.add_argument("--model", default=cfg.instruct_model_id)
    ap.add_argument("--outtype", default="q8_0",
                    help="q8_0 (default, no compiler) | f16 | q4 (needs --quantize)")
    ap.add_argument("--quantize", default=None, help="path to llama-quantize[.exe] for Q4_K_M")
    ap.add_argument("--out", default=str(RESULTS_DIR / "gguf"))
    ap.add_argument("--token", default=os.environ.get(cfg.hf_token_env))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1])
    rng = np.random.default_rng(cfg.random_seed)
    rnd = rng.standard_normal(ref.shape).astype(np.float32); rnd /= np.linalg.norm(rnd)
    vecs = {"ref": ref, "rnd": rnd, None: None}

    convert = os.path.join(args.llama, "convert_hf_to_gguf.py")
    base_outtype = "f16" if args.outtype == "q4" else args.outtype

    for tag, which, strength in CONDITIONS:
        print(f"\n== {tag} ==", flush=True)
        tmp = os.path.join(args.out, "_tmp_" + tag)
        export(args.model, vecs[which], strength, tmp, args.token)
        base = os.path.join(args.out, f"{tag}.{base_outtype}.gguf")
        subprocess.run([sys.executable, convert, tmp, "--outtype", base_outtype,
                        "--outfile", base], check=True)
        if args.outtype == "q4":
            if not args.quantize:
                print("  --outtype q4 needs --quantize; leaving f16 file.")
            else:
                q4 = os.path.join(args.out, f"{tag}.Q4_K_M.gguf")
                subprocess.run([args.quantize, base, q4, "Q4_K_M"], check=True)
                os.remove(base)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nGGUFs in {args.out}")


if __name__ == "__main__":
    main()
