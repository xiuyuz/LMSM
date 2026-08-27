#!/usr/bin/env python3
"""Prepare the three evaluation inputs from their upstream sources."""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config/evaluation.yaml"
SYSTEMS_XSTEST_SOURCE_IDS = (
    2, 4, 6, 7, 10, 11, 13, 15, 17, 18, 19, 21, 23, 24, 25,
    53, 54, 58, 59, 60, 61, 62, 64, 66, 68, 69, 71, 72, 73, 74,
    102, 103, 105, 106, 108, 109, 110, 111, 115, 116, 118, 120, 122, 124, 125,
    151, 152, 153, 154, 156, 158, 159, 161, 165, 166, 167, 170, 172, 173, 175,
    202, 203, 204, 205, 209, 210, 212, 213, 214, 216, 217, 218, 220, 222, 225,
    252, 253, 254, 255, 259, 261, 263, 264, 266, 267, 271, 272, 273, 274, 275,
    276, 278, 279, 281, 283, 284, 285, 287, 289, 290, 294, 295, 296, 297, 298,
    326, 328, 329, 333, 335, 336, 338, 340, 341, 344, 345, 346, 347, 348, 350,
    376, 377, 378, 379, 382, 385, 387, 388, 389, 393, 394, 395, 396, 399, 400,
    401, 403, 404, 406, 407, 408, 409, 411, 413, 415, 416, 419, 420, 421, 423,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_harmbench(source: dict) -> list[dict]:
    url = (
        "https://raw.githubusercontent.com/seanie12/ThinkSafe/"
        f"main/{source['file']}"
    )
    upstream = json.loads(read_url(url))
    rows = [
        {
            **dict(row),
            "id": f"harmbench_{index:03d}",
            "source": "seanie12/ThinkSafe",
            "split": source["file"],
            "source_row": index,
            "label": "harmful",
            "safety_label": "harmful",
        }
        for index, row in enumerate(upstream)
    ]
    require_count("HarmBench", rows, source["expected_rows"])
    return rows


def prepare_wildjailbreak(source: dict) -> list[dict]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise SystemExit("datasets is required to prepare WildJailbreak") from error

    dataset = load_dataset(
        source["dataset"],
        source["configuration"],
        split=source["split"],
        delimiter="\t",
        keep_default_na=False,
    )
    rows = []
    for source_row, example in enumerate(dataset):
        item = dict(example)
        if item.get("data_type") != source["data_type"]:
            continue
        index = len(rows)
        rows.append(
            {
                **item,
                "id": f"wildjailbreak_harmful_{index:04d}",
                "source": source["dataset"],
                "split": f"{source['configuration']}/{source['split']}",
                "source_row": source_row,
                "source_label": item.get("label"),
                "prompt": item["adversarial"],
                "label": "harmful",
                "safety_label": "harmful",
            }
        )
    require_count("WildJailbreak", rows, source["expected_rows"])
    return rows


def prepare_xstest(source: dict) -> list[dict]:
    url = (
        "https://raw.githubusercontent.com/paul-rottger/xstest/"
        f"main/{source['file']}"
    )
    upstream = csv.DictReader(io.StringIO(read_url(url)))
    rows = []
    for source_row, row in enumerate(upstream):
        source_label = row["label"].strip().lower()
        if source_label != source["label"]:
            continue
        index = len(rows)
        rows.append(
            {
                **row,
                "id": f"xstest_safe_{index:04d}",
                "source": "paul-rottger/xstest",
                "split": "test",
                "source_row": source_row,
                "source_id": row["id"],
                "xstest_label": source_label,
                "label": "benign",
                "safety_label": "benign",
            }
        )
    require_count("XSTest", rows, source["expected_rows"])
    return rows


def require_count(name: str, rows: list[dict], expected: int) -> None:
    if len(rows) != int(expected):
        raise ValueError(f"{name} produced {len(rows)} rows; expected {expected}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sources = config["sources"]
    xstest_rows = prepare_xstest(sources["xstest"])
    xstest_systems_ids = set(SYSTEMS_XSTEST_SOURCE_IDS)
    xstest_systems_rows = [
        row for row in xstest_rows if int(row["source_id"]) in xstest_systems_ids
    ]
    require_count("systems XSTest", xstest_systems_rows, 150)
    outputs = {
        "harmbench_602.jsonl": prepare_harmbench(sources["harmbench"]),
        "wildjailbreak_harmful_2000.jsonl": prepare_wildjailbreak(
            sources["wildjailbreak"]
        ),
        "xstest_safe_250.jsonl": xstest_rows,
        "xstest_systems_150.jsonl": xstest_systems_rows,
    }
    for filename, rows in outputs.items():
        write_jsonl(args.output_dir / filename, rows)
        print(f"wrote {len(rows)} rows to {args.output_dir / filename}")


if __name__ == "__main__":
    main()
