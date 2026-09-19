"""Batched generation for benchmark evaluation.

One-prompt-at-a-time generation wastes CPU; batching gives much larger GEMMs and
a big throughput win. Decoder-only models need LEFT padding so that every
sequence's last real token sits at the final position.
"""
from __future__ import annotations

import torch

from src.data import format_prompt


@torch.no_grad()
def batched_generate(model, tokenizer, prompts, max_new_tokens=32, batch_size=8,
                     device="cpu", is_instruct=True, log_every=0):
    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = [format_prompt(tokenizer, p, is_instruct) for p in prompts[i:i + batch_size]]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
            if log_every and (i // batch_size) % log_every == 0:
                print(f"    gen {min(i+batch_size,len(prompts))}/{len(prompts)}", flush=True)
    finally:
        tokenizer.padding_side = prev_side
    return [s.strip() for s in out]
