#!/usr/bin/env python3
"""Recompute batch-isolation margin certificates from per-pair values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "reference"
    / "batch_isolation_pair_margins.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    certified = []
    for row in rows:
        score_wave0 = float(row["score_wave0"])
        score_wave1 = float(row["score_wave1"])
        threshold = float(row["relevant_threshold"])
        distance = min(
            abs(score_wave0 - threshold),
            abs(score_wave1 - threshold),
        )
        verdict = distance > abs(score_wave0 - score_wave1)
        if verdict != bool(row["margin_certified"]):
            raise ValueError(f"stored verdict differs for {row['pair_id']}")
        certified.append(verdict)

    print(
        json.dumps(
            {
                "pair_count": len(rows),
                "decision_exact_count": sum(row["decision_exact"] for row in rows),
                "threshold_crossing_exact_count": sum(
                    row["threshold_crossing_exact"] for row in rows
                ),
                "margin_certified_count": sum(certified),
                "margin_not_certified_count": len(rows) - sum(certified),
                "margin_not_certified_ids": [
                    row["pair_id"]
                    for row, verdict in zip(rows, certified, strict=True)
                    if not verdict
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
