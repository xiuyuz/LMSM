import csv
import json

from reproduction.build_backend_results_table import build_rows, write_csv


RESULTS = (
    ("Gemma-3-4B-IT", "LMSM-SAE", "Unguarded", 122),
    ("Gemma-3-4B-IT", "LMSM-SAE", "Category-Matched Rule", 14),
    ("Gemma-3-4B-IT", "LMSM-SAE", "Six-Rule Bundle", 11),
    ("Qwen3-4B", "LMSM-Transcoder", "Unguarded", 127),
    ("Qwen3-4B", "LMSM-Transcoder", "Category-Matched Rule", 19),
    ("Qwen3-4B", "LMSM-Transcoder", "Six-Rule Bundle", 13),
)


def test_builds_backend_results_with_exact_counts_and_rates(tmp_path):
    paths = []
    for index, (model, method, configuration, unsafe_count) in enumerate(RESULTS):
        path = tmp_path / f"summary_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model": model,
                    "method": method,
                    "configuration": configuration,
                    "unsafe_count": unsafe_count,
                    "row_count": 264,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    rows = build_rows(list(reversed(paths)))

    assert [row["unsafe_count"] for row in rows] == [122, 14, 11, 127, 19, 13]
    assert all(row["denominator"] == 264 for row in rows)
    expected_rates = [
        122 / 264,
        14 / 264,
        11 / 264,
        127 / 264,
        19 / 264,
        13 / 264,
    ]
    assert [row["asr_fraction"] for row in rows] == expected_rates
    assert [row["asr_percent"] for row in rows] == [
        100.0 * rate for rate in expected_rates
    ]

    output = tmp_path / "backend_results.csv"
    write_csv(output, rows)
    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert [row["method"] for row in written] == [
        "LMSM-SAE",
        "LMSM-SAE",
        "LMSM-SAE",
        "LMSM-Transcoder",
        "LMSM-Transcoder",
        "LMSM-Transcoder",
    ]
