#!/usr/bin/env bash
# Every timing session behind Figure 7 and Section 6.5, then the results and their numbers.
set -euo pipefail
cd "$(dirname "$0")"
source ../common.sh

bash session.sh defaults "compiled_native_defaults compiled_checkpoint_graph_device_vdefaults
  compiled_anytime_graph_device_vdefaults compiled_transcoder_graph_device_vdefaults
  gemma_compiled_native_defaults gemma_compiled_sae_graph_device_vdefaults"
bash session.sh matched "compiled_native compiled_checkpoint_graph_device compiled_anytime_graph_device
  compiled_transcoder_graph_device gemma_compiled_native gemma_compiled_sae_graph_device"
bash session.sh alternatives_probes "compiled_native_defaults compiled_native eager_checkpoint_hook_host"
bash session.sh alternatives_transcoder "compiled_native_defaults compiled_transcoder_graph_host"
bash session.sh alternatives_sae "gemma_compiled_native_defaults gemma_compiled_sae_graph_host"

for size in 4 8 14 32; do
  model=Qwen/Qwen3-${size}B; [[ $size -eq 4 ]] && model=$QWEN
  util=0.85; [[ $size -eq 32 ]] && util=0.90
  profile=$("$PYTHON" synthetic_profile.py "$model" "qwen3_${size}b" "$RUNS/cost/profiles")
  MODEL=$model PROFILE=$profile UTIL=$util bash session.sh "size_${size}b" \
    "compiled_native_defaults compiled_checkpoint_graph_device_vdefaults" "32 128"
done

"$PYTHON" summarize.py
"$PYTHON" ../show.py cost
