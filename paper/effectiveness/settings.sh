# Generation settings of the effectiveness runs in Sections 6.3 and 6.4, sourced after ../common.sh.
# A later flag overrides an earlier one, so a script can append its own value to change one setting.

DECODING=(--sampling nucleus --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 16384 --max-model-len 40960
          --max-num-seqs 32 --gpu-memory-utilization 0.8 --mode compiled --observation graph --evaluation device)
MATCHED=(--max-num-batched-tokens 40960)
DEFAULTS=(--max-num-batched-tokens 32768 "${VLLM_DEFAULTS[@]}")

# run COMMAND...: run it, or print it when DRY_RUN is set.
run() {
  if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

# generate RUN_DIR MODEL PROFILE INPUT SEED [FLAGS...]: one generation run with LMSM.
generate() {
  local dir=$1 model=$2 profile=$3 input=$4 seed=$5
  shift 5
  run "$PYTHON" "$REPO/examples/run_vllm.py" --model "$model" --profile "$REPO/profiles/$profile.yaml" \
    --input "$DATA/$input.jsonl" --run-dir "$dir" --physical-gpu "$GPU" "${DECODING[@]}" --seed "$seed" "$@"
}

# judge KIND OUTPUT_DIR NAME=OUTPUTS.jsonl...: run one judge on the named arms.
judge() {
  local kind=$1 out=$2 arms=() spec
  shift 2
  for spec in "$@"; do arms+=(--arm "$spec"); done
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PAPER/judges/$kind.py" "${arms[@]}" --output-dir "$out"
}
