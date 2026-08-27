#!/bin/sh

set -eu

cd "$(dirname "$0")/.."
project_root=$(pwd)

: "${LMSM_PYTHON:=python}"
: "${LMSM_MODEL:?Set LMSM_MODEL to Qwen3-4B or a local model directory}"
: "${LMSM_DATA:?Set LMSM_DATA to the prepared benchmark directory}"
: "${LMSM_RESULTS:?Set LMSM_RESULTS to the run output directory}"

configuration="${1:-}"
physical_gpu="${2:-}"
if [ -z "$physical_gpu" ]; then
  echo "Usage: $0 CONFIGURATION PHYSICAL_GPU" >&2
  exit 2
fi

profile=profiles/lmsm_checkpoint.yaml
benchmark_mode=policy_actions
label=
max_num_seqs=32
prompt_limit=64
execution_mode=eager
policy_options=--observe-only
rule_options=

case "$configuration" in
  matched_native_start_seq1|matched_native_end_seq1)
    benchmark_mode=matched_native
    label="Matched Native"
    max_num_seqs=1
    prompt_limit=8
    policy_options=
    ;;
  matched_empty_extension_seq1)
    benchmark_mode=empty_extension
    label="Matched Empty Extension"
    max_num_seqs=1
    prompt_limit=8
    policy_options=
    ;;
  lmsm_checkpoint_seq1)
    label="LMSM-Checkpoint"
    max_num_seqs=1
    prompt_limit=8
    ;;
  lmsm_anytime_seq1)
    profile=profiles/lmsm_anytime.yaml
    label="LMSM-Anytime"
    max_num_seqs=1
    prompt_limit=8
    ;;
  matched_native_start_seq32|matched_native_end_seq32)
    benchmark_mode=matched_native
    label="Matched Native"
    policy_options=
    ;;
  compiled_native_seq1)
    benchmark_mode=matched_native
    label="Compiled Native"
    max_num_seqs=1
    prompt_limit=8
    execution_mode=compiled
    policy_options=
    ;;
  compiled_native_seq32)
    benchmark_mode=matched_native
    label="Compiled Native"
    execution_mode=compiled
    policy_options=
    ;;
  matched_empty_extension_seq32)
    benchmark_mode=empty_extension
    label="Matched Empty Extension"
    policy_options=
    ;;
  lmsm_checkpoint_seq32|lmsm_checkpoint_15rule_seq32)
    label="LMSM-Checkpoint"
    ;;
  lmsm_anytime_seq32|lmsm_anytime_15rule_seq32)
    profile=profiles/lmsm_anytime.yaml
    label="LMSM-Anytime"
    ;;
  lmsm_checkpoint_1rule_seq32)
    label="LMSM-Checkpoint"
    rule_options="--active-rule chemical_biological"
    ;;
  lmsm_checkpoint_6rule_seq32)
    label="LMSM-Checkpoint"
    rule_options="--active-rule chemical_biological --active-rule cybercrime_intrusion --active-rule copyright --active-rule misinformation_disinformation --active-rule harassment_bullying --active-rule illegal_goods_services"
    ;;
  lmsm_anytime_1rule_seq32)
    profile=profiles/lmsm_anytime.yaml
    label="LMSM-Anytime"
    rule_options="--active-rule chemical_biological"
    ;;
  lmsm_anytime_6rule_seq32)
    profile=profiles/lmsm_anytime.yaml
    label="LMSM-Anytime"
    rule_options="--active-rule chemical_biological --active-rule cybercrime_intrusion --active-rule copyright --active-rule misinformation_disinformation --active-rule harassment_bullying --active-rule illegal_goods_services"
    ;;
  *)
    echo "Unknown configuration: $configuration" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$project_root/src"

# policy_options and rule_options contain only fixed arguments selected above.
# shellcheck disable=SC2086
exec "$LMSM_PYTHON" scripts/run_vllm.py \
  --model "$LMSM_MODEL" \
  --profile "$profile" \
  --input "$LMSM_DATA/xstest_systems_150.jsonl" \
  --run-dir "$LMSM_RESULTS/systems/$configuration" \
  --physical-gpu "$physical_gpu" \
  --performance-mode \
  --benchmark-mode "$benchmark_mode" \
  --benchmark-label "$label" \
  --ignore-eos \
  --sampling greedy \
  --max-new-tokens 256 \
  --max-model-len 16384 \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.85 \
  --mode "$execution_mode" \
  --seed 0 \
  --limit "$prompt_limit" \
  --warmup-repetitions 1 \
  --measured-repetitions 5 \
  $policy_options \
  $rule_options
