#!/usr/bin/env bash
# Every mediation test behind Table 4, Figure 5 and Section 6.2, then the results and their numbers.
# The release check reads the transcoder and SAE runs of ../released_artifacts, so run those first.
set -euo pipefail
cd "$(dirname "$0")"
source ../common.sh

bash association.sh
bash ablation.sh
bash contract_tests.sh
bash not_reached.sh
"$PYTHON" summarize.py
"$PYTHON" ../show.py mediation
