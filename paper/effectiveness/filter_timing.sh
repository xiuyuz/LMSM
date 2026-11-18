#!/usr/bin/env bash
# The time column of Table 6: the time each defense adds to generating the same 128 responses.
# Run it alone on an idle GPU, after "run.sh defaults" on HarmBench, WildJailbreak and XSTest (seed 0) and
# "ARMS=disabled run.sh defaults ordinary 0". Four arms at width 32, two rounds in opposite order, one warm-up and
# three measured repetitions per arm and round:
#   native        vLLM without LMSM generates the workload, each response pinned to its recorded length
#   checkpoint    the same generation with the Checkpoint policy evaluating it
#   filter        the Llama Guard 3 filter reads the 128 Disabled responses
#   filter_after  the filter reads the same requests as Checkpoint releases them
# Usage: filter_timing.sh   DRY_RUN=1 prints the commands.
# Writes $RUNS/effectiveness/filter_timing/{workload.jsonl,workload_checkpoint.jsonl,roundR/ARM/timings.csv}.
set -euo pipefail
source "$(dirname "$0")/../common.sh"
source "$(dirname "$0")/settings.sh"

OUT=$RUNS/effectiveness/filter_timing
WL=$OUT/workload.jsonl
run "$PYTHON" "$PAPER/effectiveness/timing_workload.py" "$RUNS/effectiveness/defaults" "$OUT"

generation=(--model "$QWEN" --profile "$REPO/profiles/lmsm_checkpoint.yaml" --input "$WL" --physical-gpu "$GPU"
            --performance-mode --ignore-eos --sampling nucleus --temperature 0.6 --top-p 0.95 --top-k 20
            --max-new-tokens 16384 --max-model-len 40960 --max-num-seqs 32 --max-num-batched-tokens 32768
            --gpu-memory-utilization 0.85 --mode compiled --seed 0 --warmup-repetitions 1 --measured-repetitions 3)

arm() {  # arm NAME RUN_DIR
  case $1 in
    native)       run "$PYTHON" "$PAPER/effectiveness/pinned_generation.py" "$WL" -- "${generation[@]}" --run-dir "$2" \
                    --benchmark-mode matched_native --engine-defaults ;;
    checkpoint)   run "$PYTHON" "$PAPER/effectiveness/pinned_generation.py" "$WL" -- "${generation[@]}" --run-dir "$2" \
                    --benchmark-mode policy_actions --observe-only --observation graph --evaluation device \
                    "${VLLM_DEFAULTS[@]}" ;;
    filter)       run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PAPER/effectiveness/guard_timed.py" \
                    --workload "$WL" --run-dir "$2" ;;
    filter_after) run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PAPER/effectiveness/guard_timed.py" \
                    --workload "$OUT/workload_checkpoint.jsonl" --run-dir "$2" ;;
  esac
}

arms=(native filter filter_after checkpoint)
for round in 1 2; do
  order=("${arms[@]}")
  [[ $round == 2 ]] && order=(checkpoint filter_after filter native)
  for name in "${order[@]}"; do
    arm "$name" "$OUT/round$round/$name"
  done
done
