"""Activation-level (hook-based) interventions along the refusal direction.

Unlike the permanent weight edit in ``ablate.py`` (used to export GGUF
checkpoints), these run-time hooks modify the residual stream during the
forward pass, so several interventions can be compared on one loaded model with
one shared mechanism:

  zero     h <- h - (h.r) r                 project the component onto r to 0
  mean     h <- h - (h.r) r + mu_p r        set the component to its mean value
  reverse  h <- h - 2 (h.r) r               negate the component (anti-refusal)

``mean`` is the standard "mean ablation" control: rather than pushing the
activation off-distribution by zeroing the feature, it substitutes the feature's
typical value (its mean over harmless prompts, computed per residual point),
staying on the data manifold. Zero vs mean ablation shows whether the refusal
collapse is an artifact of off-distribution zeroing or a genuine causal effect.

Residual "points" are the raw residual stream, indexed consistently for both the
mean computation and the edit: point 0 is the input to layer 0 (the embeddings),
and point i+1 is the raw output of decoder layer i. (This deliberately avoids
``output_hidden_states``, whose final entry is post-final-norm and would not
match where the hooks act.)
"""
from __future__ import annotations

import numpy as np
import torch


def _core(model):
    return model.model if hasattr(model, "model") else model


def _unit(direction, dtype, device):
    r = torch.as_tensor(direction, dtype=dtype, device=device)
    return r / (r.norm() + 1e-8)


@torch.no_grad()
def compute_layer_mean_projections(model, tokenizer, prompts, direction,
                                   is_instruct=True, device="cpu"):
    """mu[p] = mean over all tokens of all prompts of (residual_point_p . r).

    Length is num_layers+1, using the same residual points the ablation hooks
    edit (point 0 = embeddings input, point i+1 = raw output of layer i).
    """
    from .data import format_prompt
    core = _core(model)
    r = _unit(direction, next(model.parameters()).dtype, device)
    n_points = len(core.layers) + 1
    sums = [0.0] * n_points
    counts = [0] * n_points

    def pre_hook(module, args, kwargs):
        h = args[0]
        sums[0] += float((h[0] @ r).sum()); counts[0] += h.shape[1]
        return None

    def make_out(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            sums[idx + 1] += float((h[0] @ r).sum()); counts[idx + 1] += h.shape[1]
            return None
        return hook

    handles = [core.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    for i, layer in enumerate(core.layers):
        handles.append(layer.register_forward_hook(make_out(i)))
    try:
        for p in prompts:
            text = format_prompt(tokenizer, p, is_instruct)
            enc = tokenizer(text, return_tensors="pt").to(device)
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return np.array([s / max(c, 1) for s, c in zip(sums, counts)], dtype=np.float32)


def add_ablation_hooks(model, direction, mode="zero", mu=None, device="cpu"):
    """Register hooks implementing ``mode`` along ``direction``. Returns handles.

    ``mu`` (per-point means from compute_layer_mean_projections) is required for
    mode="mean". Points: input to layer 0 (embeddings) and output of every layer.
    """
    core = _core(model)
    dtype = next(model.parameters()).dtype
    r = _unit(direction, dtype, device)
    mu_t = None if mu is None else torch.as_tensor(mu, dtype=dtype, device=device)

    def edit(h, point):
        proj = (h @ r).unsqueeze(-1)
        if mode == "zero":
            return h - proj * r
        if mode == "reverse":
            return h - 2.0 * proj * r
        if mode == "mean":
            return h - proj * r + mu_t[point] * r
        raise ValueError(mode)

    def pre_hook(module, args, kwargs):
        return (edit(args[0], 0),) + args[1:], kwargs

    def make_out(idx):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (edit(output[0], idx + 1),) + output[1:]
            return edit(output, idx + 1)
        return hook

    handles = [core.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    for i, layer in enumerate(core.layers):
        handles.append(layer.register_forward_hook(make_out(i)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()
