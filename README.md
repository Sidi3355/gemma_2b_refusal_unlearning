# Gemma-2B Refusal-Direction Ablation

An AI-safety / mechanistic-interpretability experiment: find the linear
**refusal direction** in a Gemma-2B model using a **logistic-regression linear
probe**, then **remove (negate) that direction from the model's weights** and
measure how refusal on harmful prompts changes.

The method follows the refusal-direction line of work
(Arditi et al., 2024, *Refusal in Language Models Is Mediated by a Single
Direction*). It is a standard, published interpretability technique for
studying **how** safety behaviour is represented. The code produces the
tooling and measurements; it is intended for controlled research, not for
distributing harmful content (see *Responsible use* below).

## The two models

Gemma-2B is openly available (Google's Gemma license) in a base and an
instruction-tuned variant:

| role       | Gemma 1 (default)    | Gemma 2 alternative     |
|------------|----------------------|-------------------------|
| base       | `google/gemma-2b`    | `google/gemma-2-2b`     |
| instructed | `google/gemma-2b-it` | `google/gemma-2-2b-it`  |

The refusal direction is derived from the **instruction-tuned** model (the one
that actually refuses). The Hugging Face repos are gated: accept the license on
the model page and provide a token via `HF_TOKEN` or `huggingface-cli login`.

## Method

1. **Activations.** For each harmful prompt (`data/unsafe_prompts.csv`, the 130
   provided) and each harmless prompt (`data/harmless_prompts.csv`, a matched
   control set), run a forward pass and record the residual-stream vector at the
   last prompt token, for every layer.
2. **Linear probe.** Train logistic regression to separate harmful (1) from
   harmless (0) activations. The normalised weight vector is the **refusal
   direction** `r`. The layer with the best held-out accuracy is selected
   automatically (or fix one with `--layer`).
3. **Baseline.** Generate on the 130 harmful prompts, measure the refusal rate.
4. **Weight ablation (negation).** Project `r` out of every weight matrix that
   writes into the residual stream — token embeddings, and each layer's
   attention `o_proj` and MLP `down_proj`:
   `W <- W - strength * r rᵀ W`. With `strength = 1` the model can no longer
   represent the refusal direction; this is a permanent weight edit, not a
   runtime hook. `--strength > 1` pushes toward active anti-refusal.
5. **Re-evaluate.** Generate again on the same prompts and compare refusal
   rates. Everything is saved to `results/`.

## Run

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...            # a token with Gemma access

# defaults to google/gemma-2b(-it), sweeps layers for the best probe
python run_experiment.py

# Gemma 2 2B instead, fixed layer, longer generations
python run_experiment.py --instruct google/gemma-2-2b-it \
    --probe-model google/gemma-2-2b-it --layer 12 --max-new-tokens 96
```

Outputs (`results/`): `refusal_direction_<ts>.npy` (the vector),
`experiment_<ts>.json` (baseline vs ablated refusal rates and every
per-prompt generation).

## Validate without the weights

Hugging Face access is not required to check that the pipeline is correct:

```bash
python validate_pipeline.py
```

This builds a tiny, randomly-initialised model using the **real** Gemma
architecture classes (`GemmaForCausalLM` / `Gemma2ForCausalLM`) with no
download, and runs the whole flow — activations → probe → direction → weight
ablation → generation — asserting shapes, a unit-norm direction, that exactly
the residual-writing matrices change, and that `r` is genuinely projected out
(`max |r · o_proj| ≈ 1e-9`). Random weights do not refuse, so this proves the
machinery, not the behaviour.

## Files

```
data/unsafe_prompts.csv     130 harmful prompts (provided)
data/harmless_prompts.csv   matched harmless control prompts
src/config.py               experiment configuration
src/data.py                 prompt loading + chat formatting
src/activations.py          residual-stream activation extraction
src/probe.py                linear probe -> refusal direction (+ layer sweep)
src/ablate.py               project the direction out of the weights
src/refusal_eval.py         generation + refusal-rate scoring
run_experiment.py           full experiment orchestrator
validate_pipeline.py        offline correctness check
```

## Note on this environment

This repository was developed in a sandbox whose egress policy **blocks
huggingface.co** (and Kaggle), so the real Gemma-2B weights could not be
downloaded or run here. The pipeline was instead validated offline against the
genuine Gemma architecture (see above). To produce the actual Gemma-2B numbers,
run `run_experiment.py` where Hugging Face is reachable (allowlist
`huggingface.co` in the environment's network policy, or run locally / in
Colab). GGUF files cannot be used: they are quantised for llama.cpp and do not
expose the per-layer residual stream or the individual weight matrices this
method edits — use the standard `safetensors` Hugging Face checkpoints.

## Responsible use

The purpose is to study and measure how refusal is encoded, and to demonstrate
that safety fine-tuning can be undone by a simple linear edit — a result that
motivates more robust alignment. The evaluation records refusal *rates* and
saves generations to a local file rather than printing harmful content. Use it
only on models and data you are authorised to study.
