#!/usr/bin/env python3
"""The workload of Table 6's time column.

It holds 128 real Disabled responses, 32 from each of HarmBench, WildJailbreak, XSTest and the ordinary requests,
taken from the seed-0 runs with vLLM's default optimizations. A request qualifies when its response has at least 65
generated tokens, so Checkpoint reaches its evaluation at token 64. The second file holds the same requests as
Checkpoint releases them. A response becomes the fixed refusal when the Disabled run recorded that a rule would act
at token 64, and otherwise it stays the same.

Usage: timing_workload.py DEFAULTS_RUNS_DIR OUT_DIR
"""

import json
import random
import sys
from pathlib import Path

SEED = 20261114
REFUSAL = "I can't help with that."
SOURCES = ("harmbench", "wildjailbreak", "xstest", "ordinary")


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, decisions = [], {}
    for k, bench in enumerate(SOURCES):
        with open(runs / bench / "seed_0" / "disabled" / "outputs.jsonl", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        eligible = sorted((r for r in data if len(r["ordinary_output_tokens"]) >= 65), key=lambda r: str(r["request_id"]))
        for r in random.Random(SEED + 1000 * (k + 1)).sample(eligible, 32):
            rows.append({"id": str(r["request_id"]), "prompt": r["prompt"], "response": r["released_output"],
                         "gen_tokens": len(r["ordinary_output_tokens"]), "source": bench})
            decisions[str(r["request_id"])] = any(r["policy_actions"].values())
    random.Random(SEED).shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "workload.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(out / "workload_checkpoint.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            refuses = decisions[r["id"]]
            f.write(json.dumps({**r, "response": REFUSAL if refuses else r["response"], "checkpoint_refuses": refuses},
                               ensure_ascii=False) + "\n")
    print(f"{len(rows)} requests, Checkpoint refuses {sum(decisions.values())}")


if __name__ == "__main__":
    main()
