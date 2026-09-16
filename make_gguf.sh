#!/usr/bin/env bash
# Convert the refusal-ablated Gemma model to GGUF (F16 + Q4_K_M).
# Requires llama.cpp checked out at LLAMA_DIR and its Python deps (gguf).
set -euo pipefail

MODEL_DIR="${1:-results/gemma-2b-it-refusal-ablated}"
OUT_DIR="${2:-results/gguf}"
LLAMA_DIR="${LLAMA_DIR:-/tmp/claude-0/-home-user-gemma-2b-refusal-unlearning/40a104e6-3845-520f-a4bb-e370b8d0f82d/scratchpad/llama.cpp}"
mkdir -p "$OUT_DIR"

echo "== converting $MODEL_DIR -> F16 GGUF =="
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR"     --outtype f16 --outfile "$OUT_DIR/gemma-2b-it-refusal-ablated.f16.gguf"

# Q4_K_M needs the llama-quantize binary (build once):
#   cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DLLAMA_CURL=OFF
#   cmake --build "$LLAMA_DIR/build" --target llama-quantize -j
QUANT="$LLAMA_DIR/build/bin/llama-quantize"
if [[ -x "$QUANT" ]]; then
  echo "== quantizing -> Q4_K_M =="
  "$QUANT" "$OUT_DIR/gemma-2b-it-refusal-ablated.f16.gguf"       "$OUT_DIR/gemma-2b-it-refusal-ablated.Q4_K_M.gguf" Q4_K_M
else
  echo "llama-quantize not built; only F16 GGUF produced."
fi
ls -lh "$OUT_DIR"
