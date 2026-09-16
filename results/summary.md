# Refusal-direction experiment — results

Model: `unsloth/gemma-2b-it`  (gemma-2b-it)

Refusal rate = fraction of prompts the model refused (higher = more refusals).

| Condition | Harmful prompts | Harmless prompts |
|---|---|---|
| Baseline (no edit) | 66.2% (86/130) | — |
| Zero ablation (component -> 0) | 30.0% (39/130) | — |
| Mean ablation (component -> harmless mean) | 6.9% (9/130) | — |
| Reversed (component negated) | 23.8% (31/130) | — |
| Random-direction control | 64.6% (84/130) | — |

**How to read it.** On harmful prompts, a *lower* refusal rate means the intervention removed more refusal. On harmless prompts, a *higher* refusal rate (vs baseline) means the intervention broke the model into over-refusing benign questions — the key thing to watch for the reversed model.
