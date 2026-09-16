"""Remove (negate) the refusal direction from a model's weights.

Directional ablation in weight space: every matrix that *writes* into the
residual stream has the refusal direction ``r`` projected out of its output, so
the model can no longer add anything along ``r``. Concretely, for each such
weight matrix ``W`` (mapping something -> residual) we set

    W <- W - strength * r r^T W          (output written into residual)

and for the token-embedding matrix (rows live in residual space)

    E <- E - strength * (E r) r^T

With ``strength = 1`` the component along ``r`` is fully removed. This is a
permanent edit to the weights (no runtime hooks), which is what "append the
weights with the negative of the vector / negate the refusal vector" means in
practice. ``strength > 1`` overshoots, pushing the model to actively anti-refuse.

Works for Gemma-1 (``Gemma...``) and Gemma-2 (``Gemma2...``) architectures,
whose decoder layers expose ``self_attn.o_proj`` and ``mlp.down_proj``.
"""
from __future__ import annotations

import torch


def _project_out_output(weight: torch.Tensor, r: torch.Tensor, strength: float) -> None:
    """W <- W - strength * (r r^T) W. ``weight`` is [out=hidden, in].

    Done as an in-place rank-1 update to avoid materialising the [hidden, hidden]
    ``r r^T`` matrix or a full-size product (memory-critical for the large
    matrices; a naive version can spike several GB and OOM).
    """
    rW = r @ weight                                # [in]  = r^T W
    weight.addr_(r, rW, alpha=-strength)           # W -= strength * outer(r, rW)


def _project_out_rows(weight: torch.Tensor, r: torch.Tensor, strength: float) -> None:
    """E <- E - strength * (E r) r^T. ``weight`` is [vocab, hidden].

    In-place rank-1 update; avoids allocating a [vocab, hidden] temporary
    (~2 GB for Gemma's 256k-row embedding in fp32).
    """
    coeffs = weight @ r                            # [vocab]
    weight.addr_(coeffs, r, alpha=-strength)       # E -= strength * outer(coeffs, r)


@torch.no_grad()
def ablate_refusal_direction(model, direction, strength: float = 1.0) -> dict:
    """Orthogonalise all residual-writing weights against ``direction``.

    Returns a small report of what was edited.
    """
    r = torch.as_tensor(direction, dtype=next(model.parameters()).dtype,
                         device=next(model.parameters()).device)
    r = r / (r.norm() + 1e-8)

    edited = {"embed_tokens": 0, "o_proj": 0, "down_proj": 0}

    core = model.model if hasattr(model, "model") else model

    # Token embeddings: rows are residual-space vectors.
    embed = core.embed_tokens.weight
    _project_out_rows(embed.data, r, strength)
    edited["embed_tokens"] += 1

    for layer in core.layers:
        # Attention output projection writes into the residual stream.
        w = layer.self_attn.o_proj.weight
        _project_out_output(w.data, r, strength)
        edited["o_proj"] += 1
        # MLP down projection writes into the residual stream.
        w = layer.mlp.down_proj.weight
        _project_out_output(w.data, r, strength)
        edited["down_proj"] += 1

    return edited
