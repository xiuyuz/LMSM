#!/bin/sh

set -eu

cd "$(dirname "$0")/.."
project_root=$(pwd)

: "${LMSM_PYTHON:=python}"
: "${LMSM_MODEL:?Set LMSM_MODEL to Qwen3-4B or a local model directory}"
: "${LMSM_RESULTS:?Set LMSM_RESULTS to the run output directory}"

configuration="${1:-}"
physical_gpu="${2:-}"
input_path="${3:-}"
if [ -z "$configuration" ] || [ -z "$physical_gpu" ] || [ -z "$input_path" ]; then
  echo "Usage: $0 CONFIGURATION PHYSICAL_GPU INPUT_JSONL" >&2
  exit 2
fi

case "$configuration" in
  dense_probe)
    profile=reproduction/backend_configs/cross_backend_dense_probe.yaml
    label="Task-fitted dense probe"
    ;;
  transcoder)
    profile=reproduction/backend_configs/cross_backend_transcoder.yaml
    label="Selected-coordinate transcoder"
    ;;
  *)
    echo "Unknown configuration: $configuration" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$project_root/src"

exec "$LMSM_PYTHON" scripts/run_vllm.py \
  --model "$LMSM_MODEL" \
  --profile "$profile" \
  --input "$input_path" \
  --run-dir "$LMSM_RESULTS/cross_backend/$configuration" \
  --physical-gpu "$physical_gpu" \
  --performance-mode \
  --observe-only \
  --benchmark-label "$label" \
  --ignore-eos \
  --sampling greedy \
  --max-new-tokens 64 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.8 \
  --mode eager \
  --seed-base 2000000 \
  --limit 32 \
  --warmup-repetitions 1 \
  --measured-repetitions 3
