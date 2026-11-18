#!/usr/bin/env bash
# Short answers and the not-reached outcome (Table 4, R2). Each prompt asks for an answer of one or two sentences,
# so many answers end before Checkpoint's evaluation point at token 64. The released Checkpoint policy releases such
# answers with a record, and the same policy with --not-reached-disposition deny refuses them.
# Settings as in the effectiveness runs, with vLLM's default optimizations and seed 0.
set -euo pipefail
cd "$(dirname "$0")"
source ../common.sh
OUT=$RUNS/mediation/not_reached
mkdir -p "$OUT/inputs"

for bench in harmbench_602 wildjailbreak_harmful_2000 xstest_safe_250; do
  input=$OUT/inputs/${bench}_no_think_short.jsonl
  "$PYTHON" - "$DATA/$bench.jsonl" "$input" <<'PY'
import json, sys
with open(sys.argv[2], "w") as f:
    for line in open(sys.argv[1]):
        row = json.loads(line)
        row = dict(row, prompt=str(row["prompt"]) + "\n\nAnswer in one or two sentences. /no_think", probe="no_think_short")
        f.write(json.dumps(row) + "\n")
PY
  for arm in lmsm_checkpoint:allow_with_record lmsm_checkpoint_deny:deny; do
    name=${arm%%:*}
    cmd=("$PYTHON" examples/run_vllm.py --model "$QWEN" --profile profiles/lmsm_checkpoint.yaml
         --not-reached-disposition "${arm#*:}" --input "$input" --run-dir "$OUT/$bench/$name" --physical-gpu "$GPU"
         --sampling nucleus --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 16384 --max-model-len 40960
         --max-num-seqs 32 --gpu-memory-utilization 0.8 --mode compiled --observation graph --evaluation device
         "${VLLM_DEFAULTS[@]}" --max-num-batched-tokens 32768 --seed 0)
    if [[ -n ${DRY_RUN:-} ]]; then echo "${cmd[@]}"; continue; fi
    (cd "$REPO" && VLLM_CACHE_ROOT=$RUNS/mediation/vllm_cache/not_reached "${cmd[@]}") > "$OUT/$bench.$name.log" 2>&1
  done
done
