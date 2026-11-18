#!/usr/bin/env bash
# One throughput arm: native vLLM, or LMSM evaluating its policy at each declared point without acting.
# Each request generates exactly 256 tokens with greedy decoding. One warm-up and five measured repetitions.
# Width 1 serves 8 prompts and width W serves 2W prompts.
#
# Usage: arm.sh ARM WIDTH OUT_DIR
#   ARM  compiled_native             native vLLM under matched settings
#        compiled_native_defaults    native vLLM with vLLM's default optimizations
#        compiled_POLICY_graph_EVAL  LMSM in the compiled graph, EVAL = device or host
#        eager_POLICY_hook_host      LMSM with a Python forward hook, which requires eager execution
#   POLICY is checkpoint, anytime, transcoder or sae. A suffix _vdefaults turns on vLLM's default optimizations.
#   A prefix gemma_ runs Gemma-3-4B with the SAE profile. Other arms run Qwen3-4B.
#   MODEL, PROFILE and UTIL override the model, the profile and the GPU memory share (larger models).
set -euo pipefail
source "$(dirname "$0")/../common.sh"
ARM=$1 WIDTH=$2 OUT=$3

if [[ $WIDTH -eq 1 ]]; then LIMIT=8; else LIMIT=$((2 * WIDTH)); fi
INPUT=$DATA/xstest_systems_150.jsonl
# Above 150 prompts, eight tagged copies, so that no two prompts share a prefix that prefix caching could reuse.
[[ $LIMIT -gt 150 ]] && INPUT=$RUNS/cost/inputs/xstest_systems_150_x8_tagged.jsonl

model=$QWEN; profile=profiles/lmsm_checkpoint.yaml; base=$ARM
if [[ $ARM == gemma_* ]]; then model=$GEMMA; profile=profiles/lmsm_sae.yaml; base=${ARM#gemma_}; fi
case $base in
  compiled_native)          mode=compiled; bench=matched_native; extra=() ;;
  compiled_native_defaults) mode=compiled; bench=matched_native; extra=(--engine-defaults) ;;
  *_checkpoint_*|*_anytime_*|*_transcoder_*|*_sae_*)
    mode=eager; [[ $base == compiled_* ]] && mode=compiled
    bench=policy_actions; extra=(--observe-only)
    for policy in checkpoint anytime transcoder sae; do
      [[ $base == *_${policy}_* ]] && profile=profiles/lmsm_$policy.yaml
    done
    obs=hook; [[ $base == *_graph_* ]] && obs=graph
    extra+=(--observation "$obs")
    [[ $base == *_device* ]] && extra+=(--evaluation device)
    [[ $base == *_vdefaults ]] && extra+=("${VLLM_DEFAULTS[@]}")
    ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
[[ $ARM == gemma_* ]] && extra+=(--language-model-only)

cmd=("$PYTHON" examples/run_vllm.py --model "${MODEL:-$model}" --profile "${PROFILE:-$profile}" --input "$INPUT"
     --run-dir "$OUT/${ARM}_w$WIDTH" --physical-gpu "$GPU" --performance-mode --benchmark-mode "$bench"
     --benchmark-label "$ARM" --ignore-eos --sampling greedy --max-new-tokens 256 --max-model-len 16384
     --max-num-seqs "$WIDTH" --max-num-batched-tokens 32768 --gpu-memory-utilization "${UTIL:-0.85}" --mode "$mode"
     --seed 0 --limit "$LIMIT" --warmup-repetitions 1 --measured-repetitions 5 "${extra[@]}")

if [[ ! -f $INPUT ]]; then
  mkdir -p "$(dirname "$INPUT")"
  "$PYTHON" - "$DATA/xstest_systems_150.jsonl" "$INPUT" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
with open(sys.argv[2], "w") as f:
    for k in range(8):
        for row in rows:
            copy = dict(row, id=f"{row['id']}_r{k}")
            if k:
                copy["prompt"] = f"(Request {k}) {row['prompt']}"
            f.write(json.dumps(copy) + "\n")
PY
fi
cd "$REPO"
# One compile cache per arm and model, so variants never share compiled graphs.
VLLM_CACHE_ROOT=$RUNS/cost/vllm_cache/$(basename "${MODEL:-$model}")_$ARM exec "${cmd[@]}"
