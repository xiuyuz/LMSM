#!/usr/bin/env python3
"""examples/run_vllm.py with each request's length fixed to the length of its recorded response.

With --ignore-eos every request then generates exactly its recorded number of tokens, so native vLLM and LMSM do the
same work and the filter reads the same responses. Everything else is run_vllm.py's own code.

Usage: pinned_generation.py WORKLOAD.jsonl -- RUN_VLLM_ARGUMENTS...
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import run_vllm  # noqa: E402


def main():
    split = sys.argv.index("--")
    with open(sys.argv[1], encoding="utf-8") as f:
        lengths = {str(r["id"]): int(r["gen_tokens"]) for r in map(json.loads, f)}
    original = run_vllm._sampling_params

    def pinned(components, args, request_id, seed):
        local = copy.copy(args)
        local.max_new_tokens = lengths[str(request_id)]
        return original(components, local, request_id, seed)

    run_vllm._sampling_params = pinned
    run_vllm.main(sys.argv[split + 1:])


if __name__ == "__main__":
    main()
