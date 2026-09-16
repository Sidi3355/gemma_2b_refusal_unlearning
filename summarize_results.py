#!/usr/bin/env python3
"""Summarise the experiment into human-readable tables.

Reads the latest all_conditions_harmful_*.json and all_conditions_harmless_*.json
(produced by `python run_all_conditions.py --prompts harmful|harmless`) and writes:

  results/summary.md      side-by-side refusal rates per condition + notes
  results/responses.csv   long-format per-prompt responses (open in Excel)

Run:  python summarize_results.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

from src.config import RESULTS_DIR

ORDER = ["baseline", "zero_ablation", "mean_ablation", "reversed", "random_control"]
LABEL = {
    "baseline": "Baseline (no edit)",
    "zero_ablation": "Zero ablation (component -> 0)",
    "mean_ablation": "Mean ablation (component -> harmless mean)",
    "reversed": "Reversed (component negated)",
    "random_control": "Random-direction control",
}


def _latest(pattern):
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    return json.load(open(files[-1], encoding="utf-8")) if files else None


def _rate(d, cond):
    if not d or cond not in d["conditions"]:
        return None
    return d["conditions"][cond]


def main():
    harmful = _latest("all_conditions_harmful_*.json")
    harmless = _latest("all_conditions_harmless_*.json")
    if not harmful and not harmless:
        raise SystemExit("No all_conditions_*.json found. Run run_all_conditions.py first.")

    lines = ["# Refusal-direction experiment — results", ""]
    model = (harmful or harmless)["model"]
    lines += [f"Model: `{model}`  (gemma-2b-it)", ""]
    lines += ["Refusal rate = fraction of prompts the model refused "
              "(higher = more refusals).", ""]
    lines += ["| Condition | Harmful prompts | Harmless prompts |",
              "|---|---|---|"]
    for c in ORDER:
        h = _rate(harmful, c)
        s = _rate(harmless, c)
        hcol = f"{h['refusal_rate']:.1%} ({h['refusals']}/{h['n']})" if h else "—"
        scol = f"{s['refusal_rate']:.1%} ({s['refusals']}/{s['n']})" if s else "—"
        lines.append(f"| {LABEL[c]} | {hcol} | {scol} |")
    lines += ["",
              "**How to read it.** On harmful prompts, a *lower* refusal rate means the "
              "intervention removed more refusal. On harmless prompts, a *higher* refusal "
              "rate (vs baseline) means the intervention broke the model into over-refusing "
              "benign questions — the key thing to watch for the reversed model.", ""]

    with open(RESULTS_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Long-format per-prompt CSV
    rows = []
    for setname, data in [("harmful", harmful), ("harmless", harmless)]:
        if not data:
            continue
        for cond, items in data["responses"].items():
            for it in items:
                rows.append({"prompt_set": setname, "condition": cond,
                             "refused": it["refused"], "prompt": it["prompt"],
                             "response": it["response"].replace("\n", " ")})
    with open(RESULTS_DIR / "responses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_set", "condition", "refused",
                                          "prompt", "response"])
        w.writeheader(); w.writerows(rows)

    print("\n".join(lines))
    print(f"\nwrote {RESULTS_DIR/'summary.md'} and {RESULTS_DIR/'responses.csv'} "
          f"({len(rows)} response rows)")


if __name__ == "__main__":
    main()
