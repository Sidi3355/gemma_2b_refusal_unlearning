#!/usr/bin/env python3
"""Build a llama.cpp control-vector GGUF that turns the ZERO-ablated model into
the MEAN-ablated model at inference time.

Mean ablation = zero-ablate the refusal direction, then add its per-layer mean
back. Zero ablation is baked into the model weights (gemma-2b-it-refusal-ablated
GGUF). The "add the mean back" part is exactly what a llama.cpp control vector
does: at layer il it adds a fixed vector to the residual stream. Because the
residual accumulates those additions, the per-layer vector must be the *delta*
of the mean projection, so the running sum equals the target mean at each layer.

Result runs on standard llama.cpp:
    llama-cli -m gemma-2b-it-refusal-ablated.<q>.gguf \
              --control-vector mean_ablation.controlvector.gguf -p "..."

Tensor format (matches llama.cpp's loader): general.architecture=controlvector,
controlvector.model_hint=<model arch>, controlvector.layer_count=N, tensors
"direction.1".."direction.N" (F32, [hidden]); layer index is 1-based.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import gguf

from src.config import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-hint", default="gemma",
                    help="must equal the model's general.architecture (gemma-2b -> 'gemma')")
    ap.add_argument("--out", default=str(RESULTS_DIR / "gguf" / "mean_ablation.controlvector.gguf"))
    args = ap.parse_args()

    r = np.load(sorted(glob.glob(str(RESULTS_DIR / "refusal_direction_*.npy")))[-1]).astype(np.float32)
    r /= (np.linalg.norm(r) + 1e-8)
    j = json.load(open(sorted(glob.glob(str(RESULTS_DIR / "all_conditions_harmful_*.json")))[-1]))
    mu = np.array(j["layer_mean_projections"], dtype=np.float32)   # points 0..num_layers
    n_layers = len(mu) - 1

    # Per-layer coefficients so the accumulated residual r-component equals the
    # target mean mu[il+1] at the output of layer il.
    coeffs = np.empty(n_layers, dtype=np.float32)
    coeffs[0] = mu[1]
    for il in range(1, n_layers):
        coeffs[il] = mu[il + 1] - mu[il]

    w = gguf.GGUFWriter(args.out, "controlvector")
    w.add_string("controlvector.model_hint", args.model_hint)
    w.add_uint32("controlvector.layer_count", n_layers)
    for il in range(n_layers):
        w.add_tensor(f"direction.{il + 1}", (coeffs[il] * r).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}  ({n_layers} layers, hidden={r.shape[0]}, model_hint={args.model_hint})")
    print(f"coeffs (per-layer add along r): {np.round(coeffs,3)}")


if __name__ == "__main__":
    main()
