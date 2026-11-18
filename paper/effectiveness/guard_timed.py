#!/usr/bin/env python3
"""The Llama Guard 3 filter on a timing workload, timed as examples/run_vllm.py --performance-mode times generation.

The filter reads every pair of prompt and response as guard_filter.py does. One warm-up and N measured repetitions
run over the same requests, each timed around one generate call. Each repetition uses its own cache salt, so no
repetition reuses the cached prompts of another. Model loading is not timed.

Usage: guard_timed.py --workload FILE --run-dir DIR [--max-num-seqs 32] [--measured-repetitions 3]
Writes DIR/timings.csv with one row per repetition.
"""

import argparse
import csv
import json
import time
from pathlib import Path

MODEL = "meta-llama/Llama-Guard-3-8B"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--warmup-repetitions", type=int, default=1)
    p.add_argument("--measured-repetitions", type=int, default=3)
    a = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in open(a.workload, encoding="utf-8") if line.strip()]
    chats = [tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]},
                                            {"role": "assistant", "content": r["response"]}], tokenize=False) + "\n\n"
             for r in rows]
    llm = LLM(model=MODEL, tokenizer=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=40960,
              max_num_seqs=a.max_num_seqs, max_num_batched_tokens=32768, disable_log_stats=True)
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2)

    timings = []
    for rep in range(a.warmup_repetitions + a.measured_repetitions):
        prompts = [{"prompt": chat, "cache_salt": f"{a.run_dir.resolve()}-rep{rep}"} for chat in chats]
        started = time.perf_counter()
        llm.generate(prompts, params, use_tqdm=False)
        timings.append({"repetition": rep, "is_warmup": rep < a.warmup_repetitions,
                        "generate_seconds": time.perf_counter() - started, "prompt_count": len(chats)})
    a.run_dir.mkdir(parents=True, exist_ok=True)
    with open(a.run_dir / "timings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)


if __name__ == "__main__":
    main()
