#!/usr/bin/env python3
"""Llama Guard 3 as an output filter, for Table 6.

The filter reads each pair of prompt and released response exactly as llama_guard.py does and blocks the response
when "unsafe" is more likely than "safe". A blocked response becomes LMSM's fixed refusal, so the filtered arm
differs from its input only where the filter acted. One engine reads every given run in one batch.

Usage: guard_filter.py INPUT_RUN_DIR:OUTPUT_DIR [INPUT_RUN_DIR:OUTPUT_DIR ...]
Each OUTPUT_DIR gets outputs.jsonl with the filtered responses and a "blocked" field.
"""

import json
import sys
from pathlib import Path

MODEL = "meta-llama/Llama-Guard-3-8B"
REFUSAL = "I can't help with that."
KEEP = ("request_id", "id", "prompt", "objective", "source", "label", "safety_label")


def main():
    jobs = []
    for spec in sys.argv[1:]:
        source, _, target = spec.partition(":")
        with open(Path(source) / "outputs.jsonl", encoding="utf-8") as f:
            jobs.append((Path(target), [json.loads(line) for line in f if line.strip()]))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    safe, unsafe = (tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0] for word in ("safe", "unsafe"))
    chats = [tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]},
                                            {"role": "assistant", "content": str(r["released_output"])}],
                                           tokenize=False) + "\n\n"
             for _, rows in jobs for r in rows]
    llm = LLM(model=MODEL, tokenizer=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85)
    results = iter(llm.generate(chats, SamplingParams(temperature=0.0, max_tokens=1, logprobs=2)))

    for target, rows in jobs:
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "outputs.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                logprobs = next(results).outputs[0].logprobs[0]
                blocked = (logprobs[unsafe].logprob if unsafe in logprobs else -1e5) > \
                          (logprobs[safe].logprob if safe in logprobs else -1e5)
                out = {k: r[k] for k in KEEP if k in r}
                out.update(released_output=REFUSAL if blocked else str(r["released_output"]), blocked=blocked)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
