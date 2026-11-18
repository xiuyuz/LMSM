#!/usr/bin/env bash
# Table 6: Checkpoint against a Llama Guard 3 output filter, and both together. It reads the runs of
# "run.sh defaults" on HarmBench, WildJailbreak and XSTest, so run those first.
#   1. The filter reads every Disabled response (the filtered arm) and every response that Checkpoint released
#      (the combined arm).
#   2. WildGuard judges the Disabled, Checkpoint, Anytime and filtered arms together. Llama Guard does not judge,
#      because it is the filter.
#   3. WildGuard judges the new pairs of a prompt and the fixed refusal that the combined arm releases.
# Usage: filter.sh [SEED ...]   DRY_RUN=1 prints the commands.
# Writes $RUNS/effectiveness/filter/BENCHMARK/seed_S/{filtered,combined,judge} and filter/new_pairs.
set -euo pipefail
source "$(dirname "$0")/../common.sh"
source "$(dirname "$0")/settings.sh"
[[ $# -gt 0 ]] && SEEDS=("$@")

IN=$RUNS/effectiveness/defaults
OUT=$RUNS/effectiveness/filter
disabled=() released=()
for bench in harmbench wildjailbreak xstest; do
  for seed in "${SEEDS[@]}"; do
    disabled+=("$IN/$bench/seed_$seed/disabled:$OUT/$bench/seed_$seed/filtered")
    released+=("$IN/$bench/seed_$seed/checkpoint:$OUT/$bench/seed_$seed/combined")
  done
done
run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PAPER/effectiveness/guard_filter.py" "${disabled[@]}"
run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PAPER/effectiveness/guard_filter.py" "${released[@]}"

for bench in harmbench wildjailbreak xstest; do
  for seed in "${SEEDS[@]}"; do
    i=$IN/$bench/seed_$seed o=$OUT/$bench/seed_$seed
    judge wildguard "$o/judge" disabled="$i/disabled/outputs.jsonl" checkpoint="$i/checkpoint/outputs.jsonl" \
      anytime="$i/anytime/outputs.jsonl" filtered="$o/filtered/outputs.jsonl"
  done
done
run "$PYTHON" "$PAPER/effectiveness/new_pairs.py" "$OUT"
judge wildguard "$OUT/new_pairs/judge" new_pairs="$OUT/new_pairs/outputs.jsonl"
