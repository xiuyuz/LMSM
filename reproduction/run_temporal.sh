#!/bin/sh

set -eu

cd "$(dirname "$0")/.."
project_root=$(pwd)

: "${LMSM_PYTHON:=python}"
: "${LMSM_MODEL:?Set LMSM_MODEL to Qwen3-4B or a local model directory}"
: "${LMSM_DATA:?Set LMSM_DATA to the prepared benchmark directory}"
: "${LMSM_RESULTS:?Set LMSM_RESULTS to the run output directory}"

arm=${1:-}
physical_gpu=${2:-}
if [ -z "$physical_gpu" ]; then
  echo "Usage: $0 {matched_disabled|lmsm_checkpoint|lmsm_anytime} PHYSICAL_GPU" >&2
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

export PYTHONPATH="$project_root/src"

exec "$LMSM_PYTHON" scripts/run_vllm.py \
  --model "$LMSM_MODEL" \
  --profile "$profile" \
  --input "$LMSM_DATA/wildjailbreak_harmful_2000.jsonl" \
  --run-dir "$LMSM_RESULTS/temporal/$arm" \
  --physical-gpu "$physical_gpu" \
  --sampling greedy \
  --max-new-tokens 512 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.8 \
  --mode eager \
  --seed-base 4000000 \
  $observe_flag
