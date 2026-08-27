#!/usr/bin/env python3
"""Build the SAE/transcoder backend table from judged summary files.

Each input summary is a JSON object with this compact layout::

    {
      "status": "complete",
      "model": "Gemma-3-4B-IT",
      "method": "LMSM-SAE",
      "configuration": "Unguarded",
      "unsafe_count": 122,
      "row_count": 264
    }
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLE_ORDER = (
    ("Gemma-3-4B-IT", "LMSM-SAE", "Unguarded"),
    ("Gemma-3-4B-IT", "LMSM-SAE", "Category-Matched Rule"),
    ("Gemma-3-4B-IT", "LMSM-SAE", "Six-Rule Bundle"),
    ("Qwen3-4B", "LMSM-Transcoder", "Unguarded"),
    ("Qwen3-4B", "LMSM-Transcoder", "Category-Matched Rule"),
    ("Qwen3-4B", "LMSM-Transcoder", "Six-Rule Bundle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
        help="judged summary JSON; pass once for each table row",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"summary is not complete: {path}")
    return summary


def build_rows(paths: list[Path]) -> list[dict]:
    summaries = [read_summary(path) for path in paths]
    by_key = {
        (row["model"], row["method"], row["configuration"]): row
        for row in summaries
    }
    if set(by_key) != set(TABLE_ORDER):
        raise ValueError("summaries do not contain the six expected rows")

    rows = []
    for model, method, configuration in TABLE_ORDER:
        summary = by_key[(model, method, configuration)]
        unsafe_count = int(summary["unsafe_count"])
        denominator = int(summary["row_count"])
        if denominator != 264:
            raise ValueError(f"{model} {configuration} has denominator {denominator}")
        rate = unsafe_count / denominator
        rows.append(
            {
                "model": model,
                "method": method,
                "configuration": configuration,
                "unsafe_count": unsafe_count,
                "denominator": denominator,
                "asr_fraction": rate,
                "asr_percent": 100.0 * rate,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    write_csv(args.output, build_rows(args.summary))


if __name__ == "__main__":
    main()
