#!/usr/bin/env python3
"""Build central ASR/FRR tables from completed judge summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARM_LABELS = {
    "matched_disabled": "Matched Disabled",
    "lmsm_checkpoint": "LMSM-Checkpoint",
    "lmsm_anytime": "LMSM-Anytime",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmbench-summary", type=Path, required=True)
    parser.add_argument("--wildjailbreak-summary", type=Path, required=True)
    parser.add_argument("--xstest-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise ValueError(f"summary is not complete: {path}")
    if set(value.get("arms", {})) != set(ARM_LABELS):
        raise ValueError(f"summary has unexpected arms: {path}")
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wilson(events: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = events / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def main() -> None:
    args = parse_args()
    harmbench = read_summary(args.harmbench_summary)
    wildjailbreak = read_summary(args.wildjailbreak_summary)
    xstest = read_summary(args.xstest_summary)

    central_rows = []
    interval_rows = []
    for arm, label in ARM_LABELS.items():
        hb = harmbench["arms"][arm]
        wj = wildjailbreak["arms"][arm]
        xs = xstest["arms"][arm]
        central_rows.append(
            {
                "method": label,
                "harmbench_asr_percent": 100.0 * hb["asr"],
                "harmbench_count": f"{hb['unsafe_count']}/{hb['row_count']}",
                "wildjailbreak_asr_percent": 100.0 * wj["asr"],
                "wildjailbreak_count": f"{wj['unsafe_count']}/{wj['row_count']}",
                "xstest_safe_frr_percent": 100.0 * xs["frr"],
                "xstest_count": f"{xs['refusals']}/{xs['total']}",
            }
        )
        for benchmark, metric, events, total in (
            ("HarmBench", "ASR", hb["unsafe_count"], hb["row_count"]),
            (
                "WildJailbreak",
                "ASR",
                wj["unsafe_count"],
                wj["row_count"],
            ),
            ("XSTest-safe", "FRR", xs["refusals"], xs["total"]),
        ):
            lower, upper = wilson(events, total)
            interval_rows.append(
                {
                    "benchmark": benchmark,
                    "metric": metric,
                    "method": label,
                    "events": events,
                    "denominator": total,
                    "rate_fraction": events / total,
                    "rate_percent": 100.0 * events / total,
                    "wilson95_lower_fraction": lower,
                    "wilson95_upper_fraction": upper,
                    "wilson95_lower_percent": 100.0 * lower,
                    "wilson95_upper_percent": 100.0 * upper,
                }
            )

    xstest_rows = []
    for arm, label in ARM_LABELS.items():
        row = xstest["arms"][arm]
        xstest_rows.append(
            {
                "method": label,
                "refusals": row["refusals"],
                "denominator": row["total"],
                "frr_percent": 100.0 * row["frr"],
                "actions": row["actions"],
                "action_rate_percent": 100.0 * row["action_rate"],
                "refusals_on_actioned_rows": row["refusals_on_actioned_rows"],
                "refusals_without_action": row["refusals_without_action"],
                "actioned_rows_already_refused_by_matched_disabled": row[
                    "actioned_rows_already_refused_by_matched_disabled"
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "main_safety_utility.csv", central_rows)
    write_csv(args.output_dir / "binary_rate_wilson95.csv", interval_rows)
    write_csv(args.output_dir / "xstest_decomposition.csv", xstest_rows)

    result = {
        "status": "complete",
        "harmbench_judge": {
            "model": harmbench["judge_model"],
            "protocol": harmbench["protocol"],
        },
        "wildjailbreak_judge": {
            "model": wildjailbreak["judge_model"],
            "protocol": wildjailbreak["protocol"],
        },
        "xstest_judge": {
            "model": xstest["judge_model"],
            "protocol": xstest["protocol"]["name"],
        },
        "tables": [
            "main_safety_utility.csv",
            "binary_rate_wilson95.csv",
            "xstest_decomposition.csv",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
