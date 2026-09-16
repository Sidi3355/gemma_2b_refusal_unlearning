#!/usr/bin/env bash
# Build Q4_K_M GGUFs for the four conditions that map to static weights:
#   baseline, zero_ablation, reversed, random_control.
# (mean_ablation has no faithful Gemma GGUF -- see infer_mean_ablation.py.)
#
# Requires: transformers, gguf, a built llama.cpp (convert_hf_to_gguf.py +
# llama-quantize). Set LLAMA_DIR to the llama.cpp checkout.
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:?set LLAMA_DIR to your llama.cpp checkout}"
Q="$LLAMA_DIR/build/bin/llama-quantize"
OUT="results/gguf"; mkdir -p "$OUT"
MODEL="${MODEL:-unsloth/gemma-2b-it}"
DIR_NPY="$(ls -t results/refusal_direction_*.npy | head -1)"
RAND_NPY="results/random_direction_seed0.npy"

conv () {  # <hf_dir> <tag>
  python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$1" --outtype f16 --outfile "$OUT/$2.f16.gguf"
  "$Q" "$OUT/$2.f16.gguf" "$OUT/$2.Q4_K_M.gguf" Q4_K_M && rm -f "$OUT/$2.f16.gguf"
}

# baseline: the unmodified instruct model. Export a clean copy (ensures
# tokenizer.model is present for the converter).
python3 export_ablated_model.py --strength 0.0 --out results/_baseline && \
  conv results/_baseline gemma-2b-it-baseline && rm -rf results/_baseline

# zero ablation (project refusal direction out)
python3 export_ablated_model.py --direction "$DIR_NPY" --strength 1.0 \
  --out results/_zero && conv results/_zero gemma-2b-it-refusal-ablated && rm -rf results/_zero

# reversed (negate the refusal direction)
python3 export_ablated_model.py --direction "$DIR_NPY" --strength 2.0 \
  --out results/_rev && conv results/_rev gemma-2b-it-refusal-reversed && rm -rf results/_rev

# random control (project a random direction out)
python3 export_ablated_model.py --direction "$RAND_NPY" --strength 1.0 \
  --out results/_rand && conv results/_rand gemma-2b-it-random-control && rm -rf results/_rand

ls -lh "$OUT"
sha256sum "$OUT"/*.Q4_K_M.gguf
