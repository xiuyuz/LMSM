#!/usr/bin/env bash
# Request association under churn (Table 4, R1): 14 runs of the probe policies and 8 runs of the transcoder and the
# SAE, each followed by its CPU check. Every run serves the 1,704 requests of the duplicate workload with answers of
# at most 512 tokens, so requests keep entering and leaving the batch.
set -euo pipefail
cd "$(dirname "$0")"
source ../common.sh
OUT=$RUNS/mediation/association
WORKLOAD=$RUNS/mediation/assoc_852x2.jsonl
[[ -f $WORKLOAD ]] || "$PYTHON" make_workload.py "$WORKLOAD"

run() {  # run NAME POLICY [association.py options]
  local name=$1 policy=$2; shift 2
  local model=$QWEN; [[ $policy == sae ]] && model=$GEMMA
  local cmd=("$PYTHON" association.py --model "$model" --policy "$policy" --workload "$WORKLOAD"
             --run-dir "$OUT/$name" --physical-gpu "$GPU" --max-new-tokens 512 --gpu-memory-utilization 0.3
             --mode compiled --observation graph "$@")
  if [[ -n ${DRY_RUN:-} ]]; then echo "${cmd[@]}"; return; fi
  mkdir -p "$(dirname "$OUT/$name")"
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT=$RUNS/mediation/vllm_cache/$name "${cmd[@]}" > "$OUT/$name.log" 2>&1
}
rate() {  # the staggered arrival rate: offered load 0.85 of the width, from the matching burst run's answer length
  [[ -n ${DRY_RUN:-} ]] && { echo RATE; return; }
  "$PYTHON" - "$OUT/$1" "$2" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1] + "/run/outputs.jsonl")]
mean = sum(len(r["ordinary_output_tokens"]) for r in rows) / len(rows)
print(f"{0.85 * int(sys.argv[2]) / mean:.6f}")
PY
}
# Probe policies: both schedules, device and host evaluation, widths 32 and 128, burst and staggered arrival,
# batch-invariant kernels, and chunked prefill with prefix caching.
for p in ckpt:checkpoint any:anytime; do
  short=${p%:*} policy=${p#*:}
  run probes/${short}_burst_w32_device $policy --max-num-seqs 32 --kv-cache-gib 16 --arrival burst --evaluation device
  run probes/${short}_burst_w32_host $policy --max-num-seqs 32 --kv-cache-gib 16 --arrival burst --evaluation host
  run probes/${short}_stag_w32_device $policy --max-num-seqs 32 --kv-cache-gib 16 --arrival staggered \
    --evaluation device --arrival-rate "$(rate probes/${short}_burst_w32_device 32)"
  run probes/${short}_burst_w128_device $policy --max-num-seqs 128 --kv-cache-gib 24 --arrival burst --evaluation device
  run probes/${short}_burst_w32_device_bi $policy --max-num-seqs 32 --kv-cache-gib 16 --arrival burst \
    --evaluation device --batch-invariant --attention-backend FLASH_ATTN
  chunked=(--enable-chunked-prefill --enable-prefix-caching --max-num-batched-tokens 512)
  run probes/${short}_burst_w32_device_chunk_prefix $policy --max-num-seqs 32 --kv-cache-gib 16 --arrival burst \
    --evaluation device "${chunked[@]}"
  run probes/${short}_stag_w32_device_chunk_prefix $policy --max-num-seqs 32 --kv-cache-gib 16 --arrival staggered \
    --evaluation device "${chunked[@]}" --arrival-rate "$(rate probes/${short}_burst_w32_device_chunk_prefix 32)"
done

# Transcoder (Qwen3-4B) and SAE (Gemma-3-4B) with device evaluation and vLLM's default optimizations, except that the
# engine core stays in this process. Batch budgets of 32,768 and 512 tokens, the second so that chunked prefill
# splits prompts.
for backend in transcoder sae; do
  extra=(--max-num-seqs 32 --kv-cache-gib 16 --evaluation device --async-scheduling --enable-prefix-caching
         --enable-chunked-prefill)
  [[ $backend == sae ]] && extra+=(--language-model-only)
  run $backend/burst_b32768 $backend --arrival burst --max-num-batched-tokens 32768 "${extra[@]}"
  r=$(rate $backend/burst_b32768 32)
  run $backend/staggered_b32768 $backend --arrival staggered --max-num-batched-tokens 32768 "${extra[@]}" --arrival-rate "$r"
  run $backend/burst_b512 $backend --arrival burst --max-num-batched-tokens 512 "${extra[@]}"
  run $backend/staggered_b512 $backend --arrival staggered --max-num-batched-tokens 512 "${extra[@]}" --arrival-rate "$r"
done
[[ -n ${DRY_RUN:-} ]] && exit 0

# CPU checks: summary.json (probes) and analysis_sf.json (transcoder and SAE) in each run directory.
for dir in "$OUT"/probes/*/; do "$PYTHON" check_probe_run.py "$dir"; done
for dir in "$OUT"/transcoder/*/ "$OUT"/sae/*/; do "$PYTHON" check_feature_run.py "$dir"; done
