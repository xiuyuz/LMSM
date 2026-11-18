#!/usr/bin/env python3
"""The pairs of prompt and fixed refusal that the combined row of Table 6 releases and WildGuard has not yet judged.

Each response of the combined arm keeps the WildGuard label of the same prompt and response in the four-arm pass of
filter.sh. When the filter blocks a response that Checkpoint released, the pair of the prompt and the fixed refusal
can be new. This script collects those new pairs into one file for one more WildGuard pass.

Usage: new_pairs.py FILTER_DIR   (reads FILTER_DIR/BENCHMARK/seed_S, writes FILTER_DIR/new_pairs/outputs.jsonl)
"""

import json
import sys
from pathlib import Path

DONORS = ("checkpoint", "anytime", "filtered")


def main():
    root = Path(sys.argv[1])
    new = []
    for base in sorted(root.glob("*/seed_*")):
        judged = set()
        for line in open(base / "judge" / "scored.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["arm"] in DONORS:
                judged.add((r["request_id"], r["response"]))
        for line in open(base / "combined" / "outputs.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if (str(r["request_id"]), r["released_output"]) not in judged:
                new.append({"request_id": f"{base.parent.name}/{base.name}/{r['request_id']}", "prompt": r["prompt"],
                            "released_output": r["released_output"]})
    (root / "new_pairs").mkdir(exist_ok=True)
    with open(root / "new_pairs" / "outputs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    print(f"{len(new)} new pairs")


if __name__ == "__main__":
    main()
