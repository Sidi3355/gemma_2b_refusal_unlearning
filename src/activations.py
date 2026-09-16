"""Extract residual-stream activations from a causal LM.

For each prompt we run a forward pass with ``output_hidden_states=True`` and
keep the hidden state at a chosen layer and token position. ``hidden_states``
is a tuple of length ``num_layers + 1``: index 0 is the embedding output and
index ``i`` is the residual stream after decoder layer ``i``.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import format_prompt


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    prompts: list[str],
    is_instruct: bool,
    token_position: str = "last",
    device: str = "cpu",
    batch_log_every: int = 20,
) -> np.ndarray:
    """Return an array of shape (num_prompts, num_layers+1, hidden_size).

    One summary vector per prompt per layer, taken at ``token_position``.
    """
    all_layer_vecs: list[np.ndarray] = []
    for i, prompt in enumerate(prompts):
        text = format_prompt(tokenizer, prompt, is_instruct)
        enc = tokenizer(text, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple: (L+1) x [1, seq, hidden]
        if token_position == "last":
            pos = -1
        elif token_position == "mean":
            pos = None
        else:
            raise ValueError(f"unknown token_position {token_position!r}")
        layer_vecs = []
        for h in hs:
            v = h[0].mean(dim=0) if pos is None else h[0, pos]
            layer_vecs.append(v.float().cpu().numpy())
        all_layer_vecs.append(np.stack(layer_vecs, axis=0))
        if batch_log_every and (i + 1) % batch_log_every == 0:
            print(f"    activations: {i + 1}/{len(prompts)}", flush=True)
    return np.stack(all_layer_vecs, axis=0)
