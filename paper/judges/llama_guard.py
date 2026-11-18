#!/usr/bin/env python3
"""Llama Guard 3 judge of harmful responses (HarmBench, WildJailbreak and StrongREJECT).

The judge reads each pair of prompt and released response through the guard's chat template and records the
log-probabilities of "safe" and "unsafe" as its first output token. labels.py calls a response unsafe when "unsafe"
is more likely. The rows of all arms are shuffled with a fixed seed and judged in one batch.

Usage: llama_guard.py --arm NAME=OUTPUTS.jsonl [--arm ...] --output-dir DIR [--gpu-memory-utilization 0.9]
"""

import argparse
import json
import random
from pathlib import Path

MODEL = "meta-llama/Llama-Guard-3-8B"
MISSING = -1e5  # the log-probability recorded for a token that is not among the two returned


def read_arms(specs):
    arms = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        with open(path, encoding="utf-8") as f:
            arms[name] = [json.loads(line) for line in f if line.strip()]
    ids = [[str(r["request_id"]) for r in rows] for rows in arms.values()]
    if any(x != ids[0] for x in ids):
        raise SystemExit("every arm must hold the same requests in the same order")
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=OUTPUTS.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    a = p.parse_args()

    rows = [{"arm": name, "request_id": str(r["request_id"]), "prompt": r["prompt"], "response": r["released_output"]}
            for name, arm_rows in read_arms(a.arm).items() for r in arm_rows]
    rng = random.Random(0)
    rng.shuffle(list(range(len(rows))))  # the paper's runs drew one shuffle before this one
    rng.shuffle(rows)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    safe, unsafe = (tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0] for word in ("safe", "unsafe"))
    chats = [tokenizer.apply_chat_template([{"role": "user", "content": r["prompt"]},
                                            {"role": "assistant", "content": r["response"]}], tokenize=False) + "\n\n"
             for r in rows]
    llm = LLM(model=MODEL, tokenizer=MODEL, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory_utilization)
    results = llm.generate(chats, SamplingParams(temperature=0.0, max_tokens=1, logprobs=2))

    a.output_dir.mkdir(parents=True, exist_ok=True)
    with open(a.output_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for row, result in zip(rows, results, strict=True):
            out = result.outputs[0]
            logprobs = out.logprobs[0]
            row.update(raw_judge_output=out.text,
                       safe_logprob=logprobs[safe].logprob if safe in logprobs else MISSING,
                       unsafe_logprob=logprobs[unsafe].logprob if unsafe in logprobs else MISSING)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
