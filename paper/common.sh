# Settings shared by the paper's GPU experiments. Every run script sources this file.
# Override any variable in the environment, for example QWEN=/models/Qwen3-4B.

PAPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$PAPER")"
: "${PYTHON:=python}"
: "${GPU:=0}"
: "${QWEN:=Qwen/Qwen3-4B}"
: "${GEMMA:=google/gemma-3-4b-it}"
: "${DATA:=$PAPER/data/benchmarks}"   # written by data/prepare_benchmarks.py
: "${RUNS:=$PAPER/runs}"              # every run script writes below this directory
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SEEDS=(0 1 2)

# vLLM's default optimizations: asynchronous scheduling, a background engine core, prefix caching and chunked prefill.
# Without these flags, the runs use matched settings (one synchronous process, no prefix caching, no chunked prefill).
VLLM_DEFAULTS=(--async-scheduling --engine-process mp --enable-prefix-caching --enable-chunked-prefill)
