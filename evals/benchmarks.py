"""Benchmark loaders (XSTest, HarmBench) in one common item schema.

Item schema: {"id", "prompt", "label", "group"} where
  label: "safe" | "unsafe"   (what the model *should* do)
  group: benchmark-specific subgroup (XSTest type / HarmBench semantic category)
"""
from __future__ import annotations

import csv
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def load_xstest(path: Path | None = None) -> list[dict]:
    """450 prompts: 250 safe (over-refusal probes) + 200 unsafe contrasts."""
    path = path or BENCH_DIR / "xstest_prompts.csv"
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = (r.get("label") or "").strip().lower()
            items.append({
                "id": f"xstest-{r['id']}",
                "prompt": r["prompt"].strip(),
                "label": "safe" if label == "safe" else "unsafe",
                "group": (r.get("type") or "").strip(),
            })
    return items


def load_harmbench(path: Path | None = None, category: str = "standard") -> list[dict]:
    """HarmBench behaviors. Default 'standard' (200) -- the directly-promptable
    subset. 'contextual' needs a ContextString and 'copyright' a different
    protocol, so they are excluded unless asked for.
    """
    path = path or BENCH_DIR / "harmbench_behaviors_text_all.csv"
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if category and r.get("FunctionalCategory", "").strip() != category:
                continue
            items.append({
                "id": r.get("BehaviorID") or f"hb-{len(items)}",
                "prompt": r["Behavior"].strip(),
                "label": "unsafe",
                "group": r.get("SemanticCategory", "").strip(),
            })
    return items


REGISTRY = {
    "xstest": load_xstest,
    "harmbench": load_harmbench,
}
