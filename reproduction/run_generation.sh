#!/bin/sh

set -eu

cd "$(dirname "$0")/.."
project_root=$(pwd)

: "${LMSM_PYTHON:=python}"
: "${LMSM_MODEL:?Set LMSM_MODEL to Qwen3-4B or a local model directory}"
: "${LMSM_DATA:?Set LMSM_DATA to the prepared benchmark directory}"
: "${LMSM_RESULTS:?Set LMSM_RESULTS to the run output directory}"

benchmark="${1:-}"
arm="${2:-}"
physical_gpu="${3:-}"

if [ -z "$physical_gpu" ]; then
  echo "Usage: $0 {harmbench|wildjailbreak|xstest} {matched_disabled|lmsm_checkpoint|lmsm_anytime} PHYSICAL_GPU" >&2
  exit 2
fi

case "$arm" in
  matched_disabled)
    profile=profiles/lmsm_checkpoint.yaml
    observe_flag=--observe-only
    ;;
  lmsm_checkpoint)
    profile=profiles/lmsm_checkpoint.yaml
    observe_flag=
    ;;
  lmsm_anytime)
    profile=profiles/lmsm_anytime.yaml
    observe_flag=
    ;;
  *)
    echo "Unknown arm: $arm" >&2
    exit 2
    ;;
esac

case "$benchmark" in
  harmbench)
    input="$LMSM_DATA/harmbench_602.jsonl"
    decoding="--sampling nucleus --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 16384 --max-model-len 40960 --max-num-seqs 32 --max-num-batched-tokens 40960 --gpu-memory-utilization 0.8 --mode eager --seed 0"
    ;;
  wildjailbreak)
    input="$LMSM_DATA/wildjailbreak_harmful_2000.jsonl"
    decoding="--sampling nucleus --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 16384 --max-model-len 40960 --max-num-seqs 32 --max-num-batched-tokens 40960 --gpu-memory-utilization 0.8 --mode eager --seed 0"
    ;;
  xstest)
    input="$LMSM_DATA/xstest_safe_250.jsonl"
    decoding="--sampling greedy --max-new-tokens 512 --max-model-len 16384 --max-num-seqs 32 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.8 --mode eager --seed-base 1000000"
    ;;
  *)
    echo "Unknown benchmark: $benchmark" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$project_root/src"

# The argument strings above contain only fixed options defined in this file.
# shellcheck disable=SC2086
exec "$LMSM_PYTHON" scripts/run_vllm.py \
  --model "$LMSM_MODEL" \
  --profile "$profile" \
  --input "$input" \
  --run-dir "$LMSM_RESULTS/$benchmark/$arm" \
  --physical-gpu "$physical_gpu" \
  $decoding \
  $observe_flag
