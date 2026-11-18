#!/usr/bin/env bash
# Execution and release tests (Table 4, R2 and R3) on 64 requests, under matched settings and again with vLLM's
# default optimizations and a 512-token batch budget.
#   startup              the startup check must refuse each unsupported configuration before serving
#   device_vs_host       device and host evaluation of the same requests, under matched settings only, since host
#                        evaluation refuses the default optimizations. The device run also picks the dropped requests.
#   dropped_observation  one observation dropped for 8 requests per policy, which must fail closed
#   capture_removed      the capture removed after a bypassed startup check, which must fail every request closed
set -euo pipefail
cd "$(dirname "$0")"
source ../common.sh

CENTRAL=(--sampling nucleus --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 512 --max-model-len 40960
         --max-num-seqs 32 --max-num-batched-tokens 40960 --gpu-memory-utilization 0.4 --seed 0)
HARM=(--input "$DATA/harmbench_602.jsonl" --limit 64)
# The capture-removal input: the first 32 HarmBench and the first 32 XSTest prompts.
CAPTURE_INPUT=$RUNS/mediation/contract/harmbench32_xstest32.jsonl
mkdir -p "$(dirname "$CAPTURE_INPUT")"
[[ -f $CAPTURE_INPUT ]] || cat <(head -32 "$DATA/harmbench_602.jsonl") <(head -32 "$DATA/xstest_safe_250.jsonl") > "$CAPTURE_INPUT"

for suite in matched defaults; do
  OUT=$RUNS/mediation/contract/$suite
  mkdir -p "$OUT/logs"
  engine=(); [[ $suite == defaults ]] && engine=("${VLLM_DEFAULTS[@]}" --max-num-batched-tokens 512)
  run() {  # run LOG RUN_DIR PROFILE [run_vllm.py options]. A nonzero exit is a result here, not an error.
    local log=$1 dir=$2 profile=$3; shift 3
    local cmd=(env ${RUN_ENV:-} "$PYTHON" examples/run_vllm.py --model "$QWEN" --profile "$profile"
               --run-dir "$OUT/$dir" --physical-gpu "$GPU" "${CENTRAL[@]}" "$@")
    if [[ -n ${DRY_RUN:-} ]]; then echo "${cmd[@]}"; return; fi
    (cd "$REPO" && VLLM_CACHE_ROOT=$RUNS/mediation/vllm_cache/contract_$suite "${cmd[@]}") \
      > "$OUT/logs/$log.log" 2>&1 || true
  }
  ckpt=profiles/lmsm_checkpoint.yaml

  if [[ $suite == matched ]]; then
    run startup_compiled_hook startup/compiled_hook $ckpt "${HARM[@]}" --mode compiled --observation hook
    run startup_capture_missing startup/compiled_graph_capture_missing $ckpt "${HARM[@]}" --mode compiled \
      --observation graph --evaluation device --graph-capture-disabled-for-test
    RUN_ENV=VLLM_USE_V2_MODEL_RUNNER=1 run startup_v2_runner startup/compiled_graph_v2_runner $ckpt "${HARM[@]}" \
      --mode compiled --observation graph --evaluation device
  else
    for setting in prefix:--enable-prefix-caching chunked:--enable-chunked-prefill async:--async-scheduling \
                   "mp:--engine-process mp"; do
      name=${setting%%:*}; read -r -a flag <<< "${setting#*:}"
      run startup_host_$name startup/compiled_graph_host_$name $ckpt "${HARM[@]}" --mode compiled --observation graph \
        --evaluation host "${flag[@]}"
      run startup_eager_hook_$name startup/eager_hook_$name $ckpt "${HARM[@]}" --mode eager --observation hook \
        --evaluation host "${flag[@]}"
    done
  fi

  for policy in checkpoint anytime; do
    profile=profiles/lmsm_$policy.yaml
    for ev in host device; do
      [[ $ev == host && $suite == defaults ]] && continue
      extra=(); [[ $ev == device ]] && extra=("${engine[@]}")
      run device_vs_host_${policy}_$ev device_vs_host/$policy/$ev $profile "${HARM[@]}" --mode compiled \
        --observation graph --evaluation $ev "${extra[@]}"
    done
    if [[ -n ${DRY_RUN:-} ]]; then drop=(--inject-drop-observation=REQUEST:STEP); else
      read -r -a drop <<< "$("$PYTHON" choose_drop_targets.py "$OUT/device_vs_host/$policy/device/outputs.jsonl" \
        $policy "$OUT/dropped_observation/$policy/targets.json")"
    fi
    run dropped_observation_$policy dropped_observation/$policy/run $profile "${HARM[@]}" --mode compiled \
      --observation graph --evaluation device "${engine[@]}" "${drop[@]}"
    for ev in device host; do
      [[ $ev == host && $suite == defaults ]] && continue
      extra=(); [[ $ev == device ]] && extra=("${engine[@]}")
      run capture_removed_${policy}_$ev capture_removed/$policy/$ev $profile --input "$CAPTURE_INPUT" \
        --mode compiled --observation graph --evaluation $ev --graph-capture-disabled-for-test \
        --allow-unsupported-engine "${extra[@]}"
    done
  done
done
