#!/usr/bin/env bash
# One timing session: every arm at every width, in two rounds, the second in reverse order.
# Each arm starts only after the GPU has been idle for 20 seconds. While it runs, a sampler lists the GPU's compute
# processes every 2 seconds and records any process that does not belong to the arm in foreign_processes.log.
#
# Usage: session.sh NAME "ARMS" ["WIDTHS"]      (widths default to "1 32 128")
set -euo pipefail
source "$(dirname "$0")/../common.sh"
NAME=$1 ARMS=$2 WIDTHS=${3:-1 32 128}
OUT=$RUNS/cost/$NAME
mkdir -p "$OUT/logs"
: > "$OUT/foreign_processes.log"   # a fresh log for each session

gpu_pids() { nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | tr -d ' '; }
idle_for() {  # idle_for SECONDS: the GPU runs no compute process for that long
  local t=0
  while [[ $t -lt $1 ]]; do [[ -z $(gpu_pids) ]] || return 1; sleep 2; t=$((t + 2)); done
}
belongs_to() {  # belongs_to PID ROOT: PID is ROOT or one of its descendants
  local p=$1
  while [[ -n $p && $p -gt 1 ]]; do
    [[ $p -eq $2 ]] && return 0
    p=$(ps -o ppid= -p "$p" | tr -d ' ')
  done
  return 1
}

items=()
for w in $WIDTHS; do for a in $ARMS; do items+=("$a:$w"); done; done
for round in 1 2; do
  order=("${items[@]}")
  if [[ $round -eq 2 ]]; then order=(); for ((i = ${#items[@]} - 1; i >= 0; i--)); do order+=("${items[$i]}"); done; fi
  for item in "${order[@]}"; do
    arm=${item%:*} w=${item#*:}
    until idle_for 20; do sleep 2; done
    echo "$(date '+%F %T') round $round $arm width $w"
    bash "$PAPER/cost/arm.sh" "$arm" "$w" "$OUT/round$round" > "$OUT/logs/round${round}_${arm}_w$w.log" 2>&1 &
    pid=$!
    while [[ -e /proc/$pid ]]; do
      for p in $(gpu_pids); do
        belongs_to "$p" "$pid" || echo "$(date '+%F %T') round $round $arm width $w: process $p" \
          >> "$OUT/foreign_processes.log"
      done
      sleep 2
    done
    wait "$pid"
  done
done
