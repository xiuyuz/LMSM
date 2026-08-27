#!/usr/bin/env python3
"""Summarize matched systems runs and rule-count scaling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SYSTEMS_RUNS = (
    (1, "Matched Empty Extension", "matched_empty_extension_seq1"),
    (1, "LMSM-Checkpoint", "lmsm_checkpoint_seq1"),
    (1, "LMSM-Anytime", "lmsm_anytime_seq1"),
    (32, "Matched Empty Extension", "matched_empty_extension_seq32"),
    (32, "LMSM-Checkpoint", "lmsm_checkpoint_seq32"),
    (32, "LMSM-Anytime", "lmsm_anytime_seq32"),
)

SCALING_RUNS = (
    ("Matched Empty Extension", 0, "matched_empty_extension_seq32"),
    ("LMSM-Checkpoint", 1, "lmsm_checkpoint_1rule_seq32"),
    ("LMSM-Checkpoint", 6, "lmsm_checkpoint_6rule_seq32"),
    ("LMSM-Checkpoint", 15, "lmsm_checkpoint_15rule_seq32"),
    ("LMSM-Anytime", 1, "lmsm_anytime_1rule_seq32"),
    ("LMSM-Anytime", 6, "lmsm_anytime_6rule_seq32"),
    ("LMSM-Anytime", 15, "lmsm_anytime_seq32"),
)

CHECKPOINT_REPETITION_RUNS = (
    (0, "Matched Empty Extension", "matched_empty_extension_seq32"),
    (1, "LMSM-Checkpoint", "lmsm_checkpoint_1rule_seq32"),
    (6, "LMSM-Checkpoint", "lmsm_checkpoint_6rule_seq32"),
    (15, "LMSM-Checkpoint", "lmsm_checkpoint_15rule_seq32"),
)

COMPILED_RUNS = (
    (1, "compiled_native_seq1"),
    (32, "compiled_native_seq32"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def measured_rows(run_root: Path, name: str) -> list[dict]:
    path = run_root / name / "timings.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["is_warmup"].lower() == "false"]


def summary(run_root: Path, name: str) -> dict:
    rows = measured_rows(run_root, name)
    throughput = sorted(float(row["model_tokens_per_second"]) for row in rows)
    latency = sorted(float(row["milliseconds_per_model_token"]) for row in rows)
    peak_allocated = sorted(float(row["peak_allocated_bytes"]) for row in rows)
    peak_reserved = sorted(float(row["peak_reserved_bytes"]) for row in rows)
    return {
        "measurement_count": len(rows),
        "median_model_tokens_per_second": quantile(throughput, 0.5),
        "iqr_model_tokens_per_second": (
            quantile(throughput, 0.75) - quantile(throughput, 0.25)
        ),
        "median_milliseconds_per_model_token": quantile(latency, 0.5),
        "median_peak_allocated_gib": quantile(peak_allocated, 0.5) / 2**30,
        "median_peak_reserved_gib": quantile(peak_reserved, 0.5) / 2**30,
    }


def bracketed_native_summary(run_root: Path, max_num_seqs: int) -> dict:
    start = summary(run_root, f"matched_native_start_seq{max_num_seqs}")
    end = summary(run_root, f"matched_native_end_seq{max_num_seqs}")
    throughput = 0.5 * (
        start["median_model_tokens_per_second"]
        + end["median_model_tokens_per_second"]
    )
    return {
        **start,
        "median_model_tokens_per_second": throughput,
        "median_milliseconds_per_model_token": 1000.0 / throughput,
    }


def quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    by_sequence_count = {
        (max_num_seqs, configuration): summary(args.run_root, run_name)
        for max_num_seqs, configuration, run_name in SYSTEMS_RUNS
    }
    for max_num_seqs in (1, 32):
        by_sequence_count[(max_num_seqs, "Matched Native")] = (
            bracketed_native_summary(args.run_root, max_num_seqs)
        )
    systems_rows = []
    ordered_configurations = (
        "Matched Native",
        "Matched Empty Extension",
        "LMSM-Checkpoint",
        "LMSM-Anytime",
    )
    for max_num_seqs in (1, 32):
        for configuration in ordered_configurations:
            result = by_sequence_count[(max_num_seqs, configuration)]
            native = by_sequence_count[(max_num_seqs, "Matched Native")]
            empty = by_sequence_count[(max_num_seqs, "Matched Empty Extension")]
            systems_rows.append(
                {
                    "max_num_seqs": max_num_seqs,
                    "configuration": configuration,
                    "measurement_count": result["measurement_count"],
                    "median_model_tokens_per_second": result[
                        "median_model_tokens_per_second"
                    ],
                    "iqr_model_tokens_per_second": result[
                        "iqr_model_tokens_per_second"
                    ],
                    "median_milliseconds_per_model_token": result[
                        "median_milliseconds_per_model_token"
                    ],
                    "throughput_loss_vs_matched_native_fraction": (
                        1.0
                        - result["median_model_tokens_per_second"]
                        / native["median_model_tokens_per_second"]
                    ),
                    "throughput_loss_vs_matched_empty_extension_fraction": (
                        1.0
                        - result["median_model_tokens_per_second"]
                        / empty["median_model_tokens_per_second"]
                    ),
                    "median_peak_allocated_gib": result[
                        "median_peak_allocated_gib"
                    ],
                    "median_peak_reserved_gib": result[
                        "median_peak_reserved_gib"
                    ],
                }
            )

    empty_throughput = summary(
        args.run_root, "matched_empty_extension_seq32"
    )["median_model_tokens_per_second"]
    scaling_rows = []
    for configuration, active_rules, run_name in SCALING_RUNS:
        result = summary(args.run_root, run_name)
        scaling_rows.append(
            {
                "configuration": configuration,
                "active_rules": active_rules,
                "max_num_seqs": 32,
                "measurement_count": result["measurement_count"],
                "median_model_tokens_per_second": result[
                    "median_model_tokens_per_second"
                ],
                "iqr_model_tokens_per_second": result[
                    "iqr_model_tokens_per_second"
                ],
                "throughput_loss_vs_matched_empty_extension_fraction": (
                    1.0
                    - result["median_model_tokens_per_second"] / empty_throughput
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "systems_overhead.csv", systems_rows)
    write_csv(args.output_dir / "rule_scaling.csv", scaling_rows)
    repetition_rows = []
    for active_rules, configuration, run_name in CHECKPOINT_REPETITION_RUNS:
        for row in measured_rows(args.run_root, run_name):
            repetition_rows.append(
                {
                    "active_rules": active_rules,
                    "configuration": configuration,
                    "repetition": row["repetition"],
                    "model_tokens": row["raw_output_token_count"],
                    "wall_time_seconds": row["generate_seconds"],
                    "model_tokens_per_second": row["model_tokens_per_second"],
                }
            )
    write_csv(
        args.output_dir / "checkpoint_rule_scaling_repetitions.csv",
        repetition_rows,
    )

    compiled_rows = []
    for max_num_seqs, run_name in COMPILED_RUNS:
        result = summary(args.run_root, run_name)
        eager = bracketed_native_summary(args.run_root, max_num_seqs)
        compiled_rows.append(
            {
                "max_num_seqs": max_num_seqs,
                "configuration": "Compiled Native",
                "measurement_count": result["measurement_count"],
                "median_model_tokens_per_second": result[
                    "median_model_tokens_per_second"
                ],
                "iqr_model_tokens_per_second": result[
                    "iqr_model_tokens_per_second"
                ],
                "speedup_vs_eager_native": (
                    result["median_model_tokens_per_second"]
                    / eager["median_model_tokens_per_second"]
                ),
            }
        )
    write_csv(args.output_dir / "compiled_native_ceiling.csv", compiled_rows)


if __name__ == "__main__":
    main()
