#!/usr/bin/env bash
# The second judge of Section 6.3: the official HarmBench classifier on the Disabled and Checkpoint responses of
# "run.sh defaults harmbench".
# Usage: HARMBENCH=/path/to/HarmBench second_judge.sh [SEED ...]   DRY_RUN=1 prints the commands.
set -euo pipefail
source "$(dirname "$0")/../common.sh"
source "$(dirname "$0")/settings.sh"
[[ $# -gt 0 ]] && SEEDS=("$@")

for seed in "${SEEDS[@]}"; do
  base=$RUNS/effectiveness/defaults/harmbench/seed_$seed
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PAPER/judges/harmbench_classifier.py" \
    --harmbench "${HARMBENCH:?set HARMBENCH to a HarmBench clone}" --arm disabled="$base/disabled/outputs.jsonl" \
    --arm checkpoint="$base/checkpoint/outputs.jsonl" --output-dir "$base/harmbench_classifier"
done
