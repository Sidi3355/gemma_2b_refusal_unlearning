#!/usr/bin/env bash
# Download the benchmark prompt sets (not committed: HarmBench contains harmful
# behavior prompts, and both are better fetched from their canonical source).
set -euo pipefail
mkdir -p data/benchmarks
curl -sSL -o data/benchmarks/xstest_prompts.csv \
  https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv
curl -sSL -o data/benchmarks/harmbench_behaviors_text_all.csv \
  https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv
wc -l data/benchmarks/*.csv
