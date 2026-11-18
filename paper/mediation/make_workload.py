"""Build the workload of the association runs: each of the 852 prompts of HarmBench and XSTest, submitted twice.

The 1,704 copies are placed in one seeded random order, so the two copies of a prompt arrive at unrelated positions,
see different co-runners and are usually admitted to different rows. Copy 0 is the copy submitted first.
Usage: python make_workload.py OUT.jsonl     (reads harmbench_602.jsonl and xstest_safe_250.jsonl from $DATA)
"""

import json
import os
import random
import sys
from pathlib import Path

DATA = Path(os.environ.get("DATA", Path(__file__).resolve().parent.parent / "data/benchmarks"))
SEED = 20261104


def main(out: Path):
    distinct = []
    for source, name in (("harmbench", "harmbench_602.jsonl"), ("xstest_safe", "xstest_safe_250.jsonl")):
        for line in open(DATA / name, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                distinct.append({"pair_id": str(row["id"]), "source": source,
                                 "label": row.get("safety_label", row.get("label")), "prompt": row["prompt"]})
    entries = [(index, copy) for index in range(len(distinct)) for copy in (0, 1)]
    random.Random(SEED).shuffle(entries)
    seen, rows = {}, []
    for position, (index, _) in enumerate(entries):
        copy = seen.get(index, 0)
        seen[index] = copy + 1
        src = distinct[index]
        rows.append({"id": f"{src['pair_id']}__c{copy}", "pair_id": src["pair_id"], "copy": copy,
                     "source": src["source"], "label": src["label"], "submission_position": position,
                     "prompt": src["prompt"]})
    position = {row["id"]: row["submission_position"] for row in rows}
    for row in rows:
        row["partner_submission_position"] = position[f"{row['pair_id']}__c{1 - row['copy']}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
