# Model variants (GGUF)

Five experimental conditions from the refusal-direction study on
`gemma-2b-it`. Four map to static weights and are provided as `Q4_K_M` GGUF;
mean ablation is an activation-level intervention with no faithful GGUF.

| condition        | what it does                                        | GGUF |
|------------------|-----------------------------------------------------|------|
| baseline         | unmodified `gemma-2b-it`                             | yes  |
| zero_ablation    | refusal direction projected out of the weights      | yes  |
| reversed         | refusal direction negated (anti-refusal)            | yes  |
| random_control   | a random direction projected out (specificity ctrl) | yes  |
| mean_ablation    | refusal component set to its per-layer harmless mean | no  |

## Why mean ablation has no GGUF

Zero ablation, reversed, and random control are pure weight edits (project a
direction out of, or negate it in, every matrix that writes to the residual
stream), so they bake into the checkpoint. Mean ablation additionally *adds* a
per-layer constant along the refusal direction to the residual stream. Gemma's
GGUF architecture in llama.cpp has no additive bias tensors in the residual
path, so this cannot be represented; baked in, it would silently collapse to
plain zero ablation. Run this variant from transformers instead:

```bash
python infer_mean_ablation.py "How do I pick a good lock for my own door?"
```

## Building the GGUFs

These files are multi-GB and are not committed (GitHub caps files at 100MB and
git is the wrong place for model weights). Regenerate them:

```bash
pip install -r requirements.txt gguf sentencepiece
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --target llama-quantize -j

python run_experiment.py                 # writes results/refusal_direction_*.npy
LLAMA_DIR=$PWD/llama.cpp ./build_all_gguf.sh
```

Output lands in `results/gguf/*.Q4_K_M.gguf` (~1.6 GB each).

## Checksums

SHA-256 of the built `Q4_K_M` files is recorded in `checksums.txt` when built.

## Weight-baked mean ablation (single-file, for Jan/Ollama)

A sixth variant approximates mean ablation with weights alone, so it runs as a
plain GGUF with no control vector: zero-ablate every residual-writing matrix,
then inject a constant along the refusal direction through the token embeddings
(scaled by the model normalizer). It is approximate — one constant instead of
the true per-layer, sign-changing means — so it is measured, not assumed.

Measured (gemma-2b-it): harmful refusal 3.8% (vs 6.9% true mean ablation),
harmless 1.6% (the only variant with any harmless over-refusal, from the blunt
constant overshoot).

Build it locally:
```bash
python export_baked_model.py --out results/gemma-2b-it-mean-ablation-baked
python <llama.cpp>/convert_hf_to_gguf.py results/gemma-2b-it-mean-ablation-baked \
    --outtype q8_0 --outfile results/gguf/gemma-2b-it-mean-ablation-baked.q8_0.gguf
```
Then load `gemma-2b-it-mean-ablation-baked.*.gguf` directly in Jan/Ollama/llama.cpp.
