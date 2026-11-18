#!/usr/bin/env bash
# The probe policies of Section 6.3 on Qwen3-4B: Disabled, Checkpoint and Anytime on one benchmark, then its judge.
# Usage: run.sh matched|defaults harmbench|wildjailbreak|xstest|strongreject|ordinary [SEED ...]
#   matched   matched settings (Table 5)
#   defaults  vLLM's default optimizations (StrongREJECT in Table 5, the further checks, the inputs of Table 6)
# ARMS limits the arms, for example ARMS=checkpoint. DRY_RUN=1 prints the commands.
# Writes $RUNS/effectiveness/SETTING/BENCHMARK/seed_S/{disabled,checkpoint,anytime,judge}.
set -euo pipefail
source "$(dirname "$0")/../common.sh"
source "$(dirname "$0")/settings.sh"

setting=$1 bench=$2
shift 2
[[ $# -gt 0 ]] && SEEDS=("$@")
case $setting in
  matched)  engine=("${MATCHED[@]}") ;;
  defaults) engine=("${DEFAULTS[@]}") ;;
  *) echo "unknown setting $setting" >&2; exit 2 ;;
esac
case $bench in
  harmbench)     input=harmbench_602 ;;
  wildjailbreak) input=wildjailbreak_harmful_2000 ;;
  xstest)        input=xstest_safe_250 ;;
  strongreject)  input=strongreject_313 ;;
  ordinary)      input=ordinary_3042 ;;
  *) echo "unknown benchmark $bench" >&2; exit 2 ;;
esac
read -r -a arms <<< "${ARMS:-disabled checkpoint anytime}"

for seed in "${SEEDS[@]}"; do
  base=$RUNS/effectiveness/$setting/$bench/seed_$seed
  for arm in "${arms[@]}"; do
    case $arm in
      disabled)   generate "$base/$arm" "$QWEN" lmsm_checkpoint "$input" "$seed" "${engine[@]}" --observe-only ;;
      checkpoint) generate "$base/$arm" "$QWEN" lmsm_checkpoint "$input" "$seed" "${engine[@]}" ;;
      anytime)    generate "$base/$arm" "$QWEN" lmsm_anytime "$input" "$seed" "${engine[@]}" ;;
    esac
  done
  # The judge reads the selected arms in the order of the paper's runs. Ordinary requests have no judge, since
  # the execution records show where Checkpoint intervened.
  case $bench in
    xstest)   kind=wildguard order=(disabled checkpoint anytime) ;;
    ordinary) continue ;;
    *)        kind=llama_guard order=(disabled anytime checkpoint) ;;
  esac
  specs=()
  for arm in "${order[@]}"; do
    [[ " ${arms[*]} " == *" $arm "* ]] && specs+=("$arm=$base/$arm/outputs.jsonl")
  done
  judge "$kind" "$base/judge" "${specs[@]}"
done
