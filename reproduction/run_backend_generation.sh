#!/bin/sh
set -eu

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 {sae|transcoder} {unguarded|one_rule|six_rules} SHARD PHYSICAL_GPU [RULE]" >&2
  exit 2
fi

backend=$1
configuration=$2
shard=$3
physical_gpu=$4
rule=${5:-}

: "${LMSM_BACKEND_DATA:?Set LMSM_BACKEND_DATA to the prepared backend benchmark directory}"
: "${LMSM_BACKEND_RESULTS:?Set LMSM_BACKEND_RESULTS to an output directory}"
LMSM_PYTHON=${LMSM_PYTHON:-python}

case "$backend" in
  sae)
    model=${LMSM_SAE_MODEL:-google/gemma-3-4b-it}
    profile=profiles/lmsm_sae.yaml
    thinking=
    ;;
  transcoder)
    model=${LMSM_TRANSCODER_MODEL:-Qwen/Qwen3-4B}
    profile=profiles/lmsm_transcoder.yaml
    thinking=--enable-thinking
    ;;
  *)
    echo "Unknown backend: $backend" >&2
    exit 2
    ;;
esac

case "$configuration" in
  unguarded)
    mode=unguarded
    input=$LMSM_BACKEND_DATA/harmbench_six_category_264.jsonl
    shards=2
    batch_size=32
    active_rule=
    thinking=
    output_dir=$LMSM_BACKEND_RESULTS/$backend/unguarded/shard$shard
    ;;
  one_rule)
    if [ -z "$rule" ]; then
      echo "one_rule requires a public rule ID" >&2
      exit 2
    fi
    mode=selected
    input=$LMSM_BACKEND_DATA/harmbench_by_rule/$rule.jsonl
    shards=2
    batch_size=1
    active_rule="--active-rule $rule"
    output_dir=$LMSM_BACKEND_RESULTS/$backend/one_rule/$rule/shard$shard
    ;;
  six_rules)
    mode=full
    input=$LMSM_BACKEND_DATA/harmbench_six_category_264.jsonl
    shards=12
    batch_size=1
    active_rule=
    output_dir=$LMSM_BACKEND_RESULTS/$backend/six_rules/shard$shard
    ;;
  *)
    echo "Unknown configuration: $configuration" >&2
    exit 2
    ;;
esac

if [ "$shard" -lt 0 ] || [ "$shard" -ge "$shards" ]; then
  echo "Shard $shard is outside 0..$((shards - 1))" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES=$physical_gpu "$LMSM_PYTHON" reproduction/run_hf_fixed_policy.py \
  --model "$model" \
  --profile "$profile" \
  --input "$input" \
  --output-dir "$output_dir" \
  --mode "$mode" \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --sampling nucleus \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 8192 \
  --batch-size "$batch_size" \
  --num-shards "$shards" \
  --shard-index "$shard" \
  $thinking \
  $active_rule
