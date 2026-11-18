#!/usr/bin/env bash
# What request keying and the admission reset prevent (Figure 5). Host evaluation under matched settings.
# With batch-invariant kernels, each policy and arrival pattern runs the request-keyed control, then state keyed by
# batch row with the reset at admission, then row keying without the reset. With default kernels only the controls
# run, to measure how often the two copies of a prompt decide differently.
set -euo pipefail
cd "$(dirname "$0")"
source ../common.sh
OUT=$RUNS/mediation/ablation
WORKLOAD=$RUNS/mediation/assoc_852x2.jsonl
[[ -f $WORKLOAD ]] || "$PYTHON" make_workload.py "$WORKLOAD"

run() {  # run KERNELS NAME POLICY [association.py options]
  local kernels=$1 name=$2 policy=$3; shift 3
  local extra=()
  [[ $kernels == bi ]] && extra=(--batch-invariant --attention-backend FLASH_ATTN --force-torch-topk-topp)
  local cmd=("$PYTHON" association.py --model "$QWEN" --policy "$policy" --workload "$WORKLOAD"
             --run-dir "$OUT/$kernels/$name" --physical-gpu "$GPU" --max-num-seqs 32 --max-new-tokens 512
             --gpu-memory-utilization 0.3 --kv-cache-gib 16 --mode compiled --observation graph --evaluation host
             "$@" "${extra[@]}")
  if [[ -n ${DRY_RUN:-} ]]; then echo "${cmd[@]}"; return; fi
  mkdir -p "$OUT/$kernels"
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$RUNS/mediation/vllm_cache/ablation_$kernels "${cmd[@]}" \
    > "$OUT/$kernels/$name.log" 2>&1
}
rate() {  # offered load 0.85 of width 32, from the control burst run's mean answer length
  [[ -n ${DRY_RUN:-} ]] && { echo RATE; return; }
  "$PYTHON" - "$OUT/$1" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1] + "/run/outputs.jsonl")]
mean = sum(len(r["ordinary_output_tokens"]) for r in rows) / len(rows)
print(f"{0.85 * 32 / mean:.6f}")
PY
}

for kernels in bi default; do
  for policy in checkpoint anytime; do
    for arrival in burst staggered; do
      timing=(--arrival burst)
      [[ $arrival == staggered ]] && timing=(--arrival staggered --arrival-rate "$(rate "$kernels/${policy}_control_burst")")
      run $kernels ${policy}_control_$arrival $policy "${timing[@]}"
      if [[ $kernels == bi ]]; then
        run bi ${policy}_rowkey_reset_$arrival $policy "${timing[@]}" --inject-row-keyed-state reset_on_admission
        run bi ${policy}_rowkey_noreset_$arrival $policy "${timing[@]}" --inject-row-keyed-state no_reset_on_admission
      fi
    done
  done
done
