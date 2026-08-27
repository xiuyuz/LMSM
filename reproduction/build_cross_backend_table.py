#!/usr/bin/env python3
"""Summarize the matched monitor-only cross-backend runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RUNS = (
    ("Task-fitted dense probe", "dense_probe"),
    ("Selected-coordinate transcoder", "transcoder"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def measured_timings(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["is_warmup"].lower() == "false"]


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def one_feature_score(row: dict) -> float:
    target_scores = next(iter(row["feature_maxima"].values()))
    return float(next(iter(target_scores.values())))


def summarize(run_root: Path, backend: str, directory: str) -> dict:
    run_dir = run_root / directory
    outputs = read_jsonl(run_dir / "outputs.jsonl")
    timings = measured_timings(run_dir / "timings.csv")
    scores = [one_feature_score(row) for row in outputs]
    tokens_per_second = [float(row["model_tokens_per_second"]) for row in timings]
    backend_evaluations = [float(row["encode_calls"]) for row in timings]
    token_counts = [
        len(
            row["ordinary_output_tokens"]
            if "ordinary_output_tokens" in row
            else row["raw_vllm_output_token_ids"]
        )
        for row in outputs
    ]
    return {
        "backend": backend,
        "prompts": len(outputs),
        "model_tokens_per_prompt": min(token_counts),
        "median_batched_backend_evaluations": quantile(backend_evaluations, 0.5),
        "finite_score_rows": sum(math.isfinite(score) for score in scores),
        "minimum_backend_score": min(scores),
        "maximum_backend_score": max(scores),
        "actions": sum(row["action_record"] is not None for row in outputs),
        "median_model_tokens_per_second": quantile(tokens_per_second, 0.5),
        "iqr_model_tokens_per_second": (
            quantile(tokens_per_second, 0.75) - quantile(tokens_per_second, 0.25)
        ),
    }


def main() -> None:
    args = parse_args()
    rows = [summarize(args.run_root, backend, directory) for backend, directory in RUNS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
