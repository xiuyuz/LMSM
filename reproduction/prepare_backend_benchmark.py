#!/usr/bin/env python3
"""Prepare the historical six-category HarmBench evaluation split."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DATASET = "walledai/HarmBench"
SUBSETS = ("contextual", "copyright", "standard")
SEED = 112
CATEGORY_TO_RULE = {
    "chemical_biological": "chemical_biological",
    "cybercrime_intrusion": "cybercrime_intrusion",
    "illegal": "illegal_activity",
    "misinformation_disinformation": "misinformation_disinformation",
    "book, hash_check": "book_copyright",
    "lyrics, hash_check": "lyrics_copyright",
}
CATEGORY_ORDER = (
    "book, hash_check",
    "lyrics, hash_check",
    "chemical_biological",
    "cybercrime_intrusion",
    "illegal",
    "misinformation_disinformation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def source_rows() -> list[dict]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise SystemExit("datasets is required to prepare HarmBench") from error

    rows = []
    for subset in SUBSETS:
        dataset = load_dataset(DATASET, subset, split="train").shuffle(seed=SEED)
        for source_index, example in enumerate(dataset):
            category = (
                example.get("tags")
                if subset == "copyright"
                else example.get("category")
            )
            target = example.get("context") if subset == "contextual" else None
            rows.append(
                {
                    "id": f"harmbench_{subset}_{source_index:04d}",
                    "source": DATASET,
                    "dataset_variant": "six_category_harmbench",
                    "split": "train",
                    "label": "harmful",
                    "prompt": str(example["prompt"]).strip(),
                    "goal": str(example["prompt"]).strip(),
                    "target": str(target).strip() if target else None,
                    "source_category": str(category).strip(),
                    "source_subset": subset,
                    "source_row": source_index,
                }
            )
    return rows


def held_out_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source_category"]].append(row)

    calibration_ids = set()
    for category, category_rows in sorted(grouped.items()):
        shuffled = sorted(category_rows, key=lambda row: row["id"])
        random.Random(f"{SEED}:{category}").shuffle(shuffled)
        calibration_count = max(int(len(shuffled) * 0.25 + 0.5), 5)
        calibration_count = min(calibration_count, len(shuffled) - 10)
        calibration_ids.update(row["id"] for row in shuffled[:calibration_count])

    selected = []
    for row in rows:
        source_category = row["source_category"]
        if row["id"] in calibration_ids or source_category not in CATEGORY_TO_RULE:
            continue
        selected.append(
            {
                **row,
                "split_role": "held_out_evaluation",
                "category": CATEGORY_TO_RULE[source_category],
                "matched_rule": CATEGORY_TO_RULE[source_category],
            }
        )
    if len(selected) != 264:
        raise ValueError(
            f"six-category HarmBench produced {len(selected)} rows; expected 264"
        )
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    return sorted(selected, key=lambda row: (order[row["source_category"]], row["id"]))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = held_out_rows(source_rows())
    full_path = args.output_dir / "harmbench_six_category_264.jsonl"
    write_jsonl(full_path, rows)
    print(f"wrote {len(rows)} rows to {full_path}")

    for source_category in CATEGORY_ORDER:
        rule_id = CATEGORY_TO_RULE[source_category]
        selected = [row for row in rows if row["matched_rule"] == rule_id]
        path = args.output_dir / "harmbench_by_rule" / f"{rule_id}.jsonl"
        write_jsonl(path, selected)
        print(f"wrote {len(selected)} rows to {path}")


if __name__ == "__main__":
    main()
