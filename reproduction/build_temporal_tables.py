#!/usr/bin/env python3
"""Build trigger timing tables from the matched 512-token runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disabled-outputs", type=Path, required=True)
    parser.add_argument("--checkpoint-actions", type=Path, required=True)
    parser.add_argument("--anytime-actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_id(row: dict) -> str:
    return str(row.get("request_id") or row.get("dataset_id") or row["id"])


def ordinary_tokens(row: dict) -> list[int]:
    if "ordinary_output_tokens" in row:
        return row["ordinary_output_tokens"]
    return row["raw_vllm_output_token_ids"]


def action_category(row: dict) -> str:
    return str(row.get("target_id") or row.get("category") or row["rule_id"])


def distribution(values: list[int]) -> dict:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "q1": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.5),
        "q3": quantile(ordered, 0.75),
        "p90": quantile(ordered, 0.9),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline = {
        row_id(row): len(ordinary_tokens(row))
        for row in read_jsonl(args.disabled_outputs)
    }
    arms = (
        ("LMSM-Checkpoint", read_jsonl(args.checkpoint_actions)),
        ("LMSM-Anytime", read_jsonl(args.anytime_actions)),
    )

    cdf_rows = []
    distribution_rows = []
    prefix_rows = []
    for arm, actions in arms:
        steps = [int(row["step"]) for row in actions]
        counts = Counter(steps)
        cumulative = 0
        for step in sorted(counts):
            cumulative += counts[step]
            cdf_rows.append(
                {
                    "arm": arm,
                    "category": "all",
                    "denominator": len(steps),
                    "step": step,
                    "cumulative_count": cumulative,
                    "fraction": cumulative / len(steps),
                }
            )

        grouped = defaultdict(list)
        for row in actions:
            grouped[action_category(row)].append(int(row["step"]))
        distribution_rows.append(
            {
                "arm": arm,
                "metric": "trigger_step",
                "category": "all",
                **distribution(steps),
            }
        )
        for category in sorted(grouped):
            distribution_rows.append(
                {
                    "arm": arm,
                    "metric": "trigger_step",
                    "category": category,
                    **distribution(grouped[category]),
                }
            )

        admitted = [len(row["admitted_prefix_tokens"]) for row in actions]
        avoided = [
            max(baseline[row_id(row)] - len(row["admitted_prefix_tokens"]), 0)
            for row in actions
        ]
        prefix_rows.append(
            {
                "arm": arm,
                "metric": "admitted_prefix_tokens",
                "category": "all",
                **distribution(admitted),
            }
        )
        prefix_rows.append(
            {
                "arm": arm,
                "metric": "avoided_model_tokens",
                "category": "all",
                **distribution(avoided),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trigger_cdf.csv", cdf_rows)
    write_csv(
        args.output_dir / "trigger_distributions.csv",
        distribution_rows + prefix_rows,
    )


if __name__ == "__main__":
    main()
