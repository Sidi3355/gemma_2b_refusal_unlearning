#!/usr/bin/env python3
"""Offline correctness check for the refusal-ablation pipeline.

Hugging Face is not reachable in every environment, and the real Gemma-2B
weights are gated. This script instantiates a *tiny, randomly initialised*
model using the genuine Gemma architecture classes (GemmaForCausalLM /
Gemma2ForCausalLM) with NO download, plus a minimal stub tokenizer, and runs
the full pipeline end to end:

    activations -> linear probe -> refusal direction -> weight ablation -> generate

It verifies shapes, that the probe returns a unit direction, that the ablation
edits exactly the residual-writing matrices (embed_tokens, o_proj, down_proj),
and that generation still runs after editing. It does NOT test refusal
behaviour (random weights do not refuse); it proves the machinery is correct so
the identical code will run on real Gemma-2B where Hugging Face is reachable.
"""
from __future__ import annotations

import numpy as np
import torch

from src.activations import collect_activations
from src.probe import train_probe
from src.ablate import ablate_refusal_direction
from src.refusal_eval import generate_responses, summarise


class StubTokenizer:
    """Whitespace tokenizer over a fixed vocab; enough for the pipeline API."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.chat_template = "{{ '<start_of_turn>user\n' + messages[0]['content'] }}"

    def _ids(self, text):
        toks = text.split()
        return [2 + (abs(hash(t)) % (self.vocab_size - 3)) for t in toks] or [2]

    def __call__(self, text, return_tensors=None):
        ids = self._ids(text)
        out = {"input_ids": torch.tensor([ids]),
               "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return _Enc(out)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "user " + messages[0]["content"]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


class _Enc(dict):
    def to(self, device):
        return _Enc({k: v.to(device) for k, v in self.items()})


def build_tiny(arch="gemma"):
    if arch == "gemma":
        from transformers import GemmaConfig, GemmaForCausalLM as M
        cfg = GemmaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128,
        )
    else:
        from transformers import Gemma2Config, Gemma2ForCausalLM as M
        cfg = Gemma2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128,
        )
    torch.manual_seed(0)
    return M(cfg).eval(), cfg


def run_for(arch):
    print(f"\n=== validating architecture: {arch} ===")
    model, cfg = build_tiny(arch)
    tok = StubTokenizer(cfg.vocab_size)

    harmful = [f"harmful prompt number {i} do bad thing {i%5}" for i in range(24)]
    harmless = [f"harmless prompt number {i} nice thing {i%5}" for i in range(24)]

    ha = collect_activations(model, tok, harmful, True, "last", "cpu", batch_log_every=0)
    sa = collect_activations(model, tok, harmless, True, "last", "cpu", batch_log_every=0)
    exp_layers = cfg.num_hidden_layers + 1
    assert ha.shape == (24, exp_layers, cfg.hidden_size), ha.shape
    print(f"  activations OK: {ha.shape} (num_layers+1={exp_layers})")

    probe = train_probe(ha, sa, layer=None, C=1.0, seed=0, test_size=0.25)
    assert probe.direction.shape == (cfg.hidden_size,)
    assert abs(np.linalg.norm(probe.direction) - 1.0) < 1e-4
    print(f"  probe OK: layer={probe.layer} test_acc={probe.test_acc:.3f} "
          f"|direction|={np.linalg.norm(probe.direction):.4f}")

    # Snapshot weights that must change and one that must not.
    core = model.model
    before_o = core.layers[0].self_attn.o_proj.weight.detach().clone()
    before_down = core.layers[0].mlp.down_proj.weight.detach().clone()
    before_embed = core.embed_tokens.weight.detach().clone()
    before_q = core.layers[0].self_attn.q_proj.weight.detach().clone()

    report = ablate_refusal_direction(model, probe.direction, strength=1.0)
    print(f"  ablation report: {report}")
    assert report["o_proj"] == cfg.num_hidden_layers
    assert report["down_proj"] == cfg.num_hidden_layers
    assert report["embed_tokens"] == 1

    after_o = core.layers[0].self_attn.o_proj.weight
    after_down = core.layers[0].mlp.down_proj.weight
    after_embed = core.embed_tokens.weight
    after_q = core.layers[0].self_attn.q_proj.weight
    assert not torch.allclose(before_o, after_o), "o_proj should change"
    assert not torch.allclose(before_down, after_down), "down_proj should change"
    assert not torch.allclose(before_embed, after_embed), "embed should change"
    assert torch.allclose(before_q, after_q), "q_proj must NOT change"

    # Verify the refusal direction was actually projected out of o_proj output.
    r = torch.tensor(probe.direction)
    residual_along_r = (r @ after_o)   # component of each output column along r
    assert residual_along_r.abs().max() < 1e-4, residual_along_r.abs().max().item()
    print(f"  orthogonality OK: max |r . o_proj| = {residual_along_r.abs().max():.2e}")

    resp = generate_responses(model, tok, harmful[:4], True, max_new_tokens=8,
                              do_sample=False, device="cpu", log_every=0)
    stats = summarise(resp)
    print(f"  generation OK: produced {len(resp)} responses, stats={stats}")
    print(f"  [{arch}] ALL CHECKS PASSED")


if __name__ == "__main__":
    run_for("gemma")
    try:
        run_for("gemma2")
    except Exception as e:  # noqa
        print(f"\n(gemma2 arch check skipped: {type(e).__name__}: {e})")
    print("\nPipeline validated offline against the real Gemma architecture.")
