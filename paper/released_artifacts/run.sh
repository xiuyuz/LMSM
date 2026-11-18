#!/usr/bin/env bash
# The released interpretability artifacts of Section 6.4: a transcoder on Qwen3-4B and an SAE on Gemma-3-4B.
# Usage: run.sh EXPERIMENT [SEED ...]   DRY_RUN=1 prints the commands.
#   transcoder_matched   the transcoder policy under matched settings, on HarmBench and XSTest
#   transcoder_defaults  the transcoder policy with vLLM's default optimizations
#   sae_defaults         the SAE policy and its Disabled arm, with vLLM's default optimizations
#   combined             probe rules and transcoder rules as one policy, and its Disabled arm, under matched settings
#                        (the composed policy evaluates on the host, so it runs with matched settings only)
#   copyright            Disabled answers to the 100 copyright requests, and the two copyright rules of each
#                        artifact deployed on those requests and on XSTest (HARMBENCH must name a HarmBench clone)
# Writes $RUNS/released_artifacts/EXPERIMENT/...
set -euo pipefail
source "$(dirname "$0")/../common.sh"
source "$PAPER/effectiveness/settings.sh"

experiment=$1
shift
[[ $# -gt 0 ]] && SEEDS=("$@")
OUT=$RUNS/released_artifacts/$experiment
GEMMA_ARGS=(--language-model-only)

for seed in "${SEEDS[@]}"; do
  case $experiment in
    transcoder_matched|transcoder_defaults)
      engine=("${MATCHED[@]}" --gpu-memory-utilization 0.4)
      [[ $experiment == transcoder_defaults ]] && engine=("${DEFAULTS[@]}")
      for bench in harmbench xstest; do
        input=harmbench_602; [[ $bench == xstest ]] && input=xstest_safe_250
        base=$OUT/$bench/seed_$seed
        generate "$base/transcoder" "$QWEN" lmsm_transcoder "$input" "$seed" "${engine[@]}"
      done
      judge llama_guard "$OUT/harmbench/seed_$seed/judge" transcoder="$OUT/harmbench/seed_$seed/transcoder/outputs.jsonl"
      judge wildguard "$OUT/xstest/seed_$seed/judge" transcoder="$OUT/xstest/seed_$seed/transcoder/outputs.jsonl" ;;
    sae_defaults)
      for bench in harmbench xstest; do
        input=harmbench_602; [[ $bench == xstest ]] && input=xstest_safe_250
        base=$OUT/$bench/seed_$seed
        generate "$base/disabled" "$GEMMA" lmsm_sae "$input" "$seed" "${DEFAULTS[@]}" "${GEMMA_ARGS[@]}" --observe-only
        generate "$base/sae" "$GEMMA" lmsm_sae "$input" "$seed" "${DEFAULTS[@]}" "${GEMMA_ARGS[@]}"
      done
      base=$OUT/harmbench/seed_$seed
      judge llama_guard "$base/judge" disabled="$base/disabled/outputs.jsonl" sae="$base/sae/outputs.jsonl"
      base=$OUT/xstest/seed_$seed
      judge wildguard "$base/judge" disabled="$base/disabled/outputs.jsonl" sae="$base/sae/outputs.jsonl" ;;
    combined)
      # The KV cache is fixed at the size of the probe policies' matched runs.
      engine=("${MATCHED[@]}" --gpu-memory-utilization 0.3 --kv-cache-gib 63.70751953125 --evaluation host)
      for bench in harmbench xstest; do
        input=harmbench_602; [[ $bench == xstest ]] && input=xstest_safe_250
        base=$OUT/$bench/seed_$seed
        generate "$base/all" "$QWEN" lmsm_composed_all "$input" "$seed" "${engine[@]}"
        generate "$base/shared" "$QWEN" lmsm_composed_shared "$input" "$seed" "${engine[@]}"
        generate "$base/disabled" "$QWEN" lmsm_composed_all "$input" "$seed" "${engine[@]}" --observe-only
      done
      base=$OUT/harmbench/seed_$seed
      judge llama_guard "$base/judge" disabled="$base/disabled/outputs.jsonl" shared="$base/shared/outputs.jsonl" \
        all="$base/all/outputs.jsonl"
      base=$OUT/xstest/seed_$seed
      judge wildguard "$base/judge" disabled="$base/disabled/outputs.jsonl" all="$base/all/outputs.jsonl" \
        shared="$base/shared/outputs.jsonl" ;;
    copyright)
      t=$OUT/transcoder s=$OUT/sae
      generate "$t/requests/seed_$seed/disabled" "$QWEN" lmsm_transcoder copyright_100 "$seed" "${MATCHED[@]}" --observe-only
      generate "$t/requests/seed_$seed/policy" "$QWEN" lmsm_transcoder_copyright copyright_100 "$seed" "${MATCHED[@]}"
      generate "$t/xstest/seed_$seed/disabled" "$QWEN" lmsm_transcoder_copyright xstest_safe_250 "$seed" "${MATCHED[@]}" \
        --observe-only
      generate "$t/xstest/seed_$seed/policy" "$QWEN" lmsm_transcoder_copyright xstest_safe_250 "$seed" "${MATCHED[@]}"
      generate "$s/requests/seed_$seed/disabled" "$GEMMA" lmsm_sae copyright_100 "$seed" "${DEFAULTS[@]}" "${GEMMA_ARGS[@]}" \
        --observe-only
      generate "$s/requests/seed_$seed/policy" "$GEMMA" lmsm_sae_copyright copyright_100 "$seed" "${DEFAULTS[@]}" \
        "${GEMMA_ARGS[@]}"
      generate "$s/xstest/seed_$seed/policy" "$GEMMA" lmsm_sae_copyright xstest_safe_250 "$seed" "${DEFAULTS[@]}" \
        "${GEMMA_ARGS[@]}"
      judge wildguard "$t/xstest/seed_$seed/judge" disabled="$t/xstest/seed_$seed/disabled/outputs.jsonl" \
        policy="$t/xstest/seed_$seed/policy/outputs.jsonl"
      judge wildguard "$s/xstest/seed_$seed/judge" policy="$s/xstest/seed_$seed/policy/outputs.jsonl"
      run "$PYTHON" "$PAPER/judges/copyright_classifier.py" --harmbench "${HARMBENCH:?set HARMBENCH to a HarmBench clone}" \
        --inputs "$DATA/copyright_100.jsonl" --run "$t/requests/seed_$seed/disabled" --run "$s/requests/seed_$seed/disabled" ;;
    *) echo "unknown experiment $experiment" >&2; exit 2 ;;
  esac
done
