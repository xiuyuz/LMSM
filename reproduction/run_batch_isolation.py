#!/usr/bin/env python3
"""Reproduce the 64-request continuous-batch isolation experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SCRIPTS = PROJECT_ROOT / "scripts"
HARMFUL_SOURCE_IDS = (
    2, 3, 10, 12, 17, 18, 25, 28,
    30, 33, 42, 43, 52, 54, 72, 74,
)
BENIGN_SOURCE_IDS = (
    9, 12, 14, 16, 20, 22, 51, 52,
    55, 56, 57, 63, 65, 67, 70, 75,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--xstest-input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_inputs(xstest_input: Path) -> list[dict]:
    sys.path.insert(0, str(HERE))
    from prepare_backend_benchmark import source_rows

    harmful_by_id = {row["id"]: row for row in source_rows()}
    benign_by_id = {
        int(row["source_id"]): row for row in read_jsonl(xstest_input)
    }
    harmful = [
        harmful_by_id[f"harmbench_contextual_{source_id:04d}"]
        for source_id in HARMFUL_SOURCE_IDS
    ]
    benign = [benign_by_id[source_id] for source_id in BENIGN_SOURCE_IDS]

    prepared = []
    caps = (8, 16, 24, 32)
    for wave in (0, 1):
        for index, source in enumerate(harmful + benign):
            kind = "harmful" if index < len(harmful) else "benign"
            pair_id = f"batch_isolation_{kind}_{index % 16:02d}"
            prepared.append(
                {
                    "id": f"{pair_id}_wave{wave}",
                    "pair_id": pair_id,
                    "wave": wave,
                    "prompt": source["prompt"],
                    "request_max_new_tokens": caps[index % len(caps)],
                }
            )
    return prepared


def load_thresholds(profile_path: Path) -> dict[str, float]:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    bindings = {item["id"]: item for item in profile["backend_bindings"]}
    rules = {item["id"]: item for item in profile["rule_library"]["rules"]}
    reports = {}
    thresholds = {}
    for rule_id in profile["policy"]["active_rules"]:
        rule = rules[rule_id]
        binding = bindings[rule["binding"]]
        report_path = Path(binding["report_path"])
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        if report_path not in reports:
            reports[report_path] = json.loads(
                report_path.read_text(encoding="utf-8")
            )
        report = reports[report_path]
        calibration_key = rule["condition"]["calibration_key"]
        thresholds[rule_id] = float(report["probes"][calibration_key]["threshold"])
    return thresholds


class BatchEventRecorder:
    def __init__(self, state, request_id_key: str):
        self.state = state
        self.request_id_key = request_id_key
        self.events = []
        self.slot_history = {}
        self.original_update = state.update_batch

    def update(self, batch_update):
        if batch_update is None:
            return self.original_update(batch_update)

        before = {row: request.request_id for row, request in self.state.rows.items()}
        removed = [
            {"row": int(row), "request_id": before.get(int(row))}
            for row in batch_update.removed
        ]
        added = []
        for row, params, _prompt_ids, _output_ids in batch_update.added:
            row = int(row)
            request_id = str(params.extra_args[self.request_id_key])
            history = self.slot_history.get(row, [])
            added.append(
                {
                    "row": row,
                    "request_id": request_id,
                    "slot_reused": bool(history and history[-1] != request_id),
                }
            )
        moved = [
            {
                "source": int(source),
                "destination": int(destination),
                "direction": direction.name,
            }
            for source, destination, direction in batch_update.moved
        ]

        self.original_update(batch_update)
        initialized_empty = {}
        evaluator = self.state.evaluator
        for item in added:
            request_id = item["request_id"]
            initialized_empty[request_id] = {
                "features_empty": evaluator.feature_maxima_for(request_id) == {},
                "scores_empty": evaluator.scores_for(request_id) == {},
                "score_steps_empty": evaluator.score_steps_for(request_id) == {},
                "actions_empty": evaluator.actions_for(request_id) == {},
            }
            self.slot_history.setdefault(item["row"], []).append(request_id)

        if added or removed or moved:
            self.events.append(
                {
                    "batch_size": int(batch_update.batch_size),
                    "removed": removed,
                    "added": added,
                    "moved": moved,
                    "initialized_empty": initialized_empty,
                }
            )


def decision(row: dict) -> dict:
    action = row["action_record"]
    return {
        "action": action["action"] if action else None,
        "category": action["target_id"] if action else None,
        "step": action["step"] if action else None,
    }


def analyze(
    prepared: list[dict],
    outputs: list[dict],
    recorder,
    thresholds: dict[str, float],
) -> dict:
    output_ids = [row["request_id"] for row in outputs]
    expected_ids = [row["id"] for row in prepared]
    by_id = {row["request_id"]: row for row in outputs}
    pair_results = []
    for pair_id in [row["pair_id"] for row in prepared if row["wave"] == 0]:
        left = by_id[f"{pair_id}_wave0"]
        right = by_id[f"{pair_id}_wave1"]
        score_keys = sorted(set(left["policy_scores"]) | set(right["policy_scores"]))
        max_score_difference = max(
            (
                abs(left["policy_scores"][key] - right["policy_scores"][key])
                for key in score_keys
            ),
            default=0.0,
        )
        left_decision = decision(left)
        right_decision = decision(right)
        if left_decision["category"] is not None:
            relevant_rule = left_decision["category"]
            relevant_rule_mode = "selected_action_rule"
        else:
            relevant_rule = min(
                score_keys,
                key=lambda key: min(
                    abs(left["policy_scores"][key] - thresholds[key]),
                    abs(right["policy_scores"][key] - thresholds[key]),
                ),
            )
            relevant_rule_mode = "nearest_rule_for_nonaction"

        threshold = thresholds[relevant_rule]
        score_wave0 = left["policy_scores"][relevant_rule]
        score_wave1 = right["policy_scores"][relevant_rule]
        threshold_margin_wave0 = score_wave0 - threshold
        threshold_margin_wave1 = score_wave1 - threshold
        score_difference = abs(score_wave0 - score_wave1)
        minimum_distance = min(
            abs(threshold_margin_wave0),
            abs(threshold_margin_wave1),
        )
        # Observed-difference certificate: min_w |score_w - threshold| > |score_0 - score_1|.
        pair_results.append(
            {
                "pair_id": pair_id,
                "decision_wave0": left_decision,
                "decision_wave1": right_decision,
                "decision_exact": left_decision == right_decision,
                "threshold_crossing_exact": (
                    left["policy_actions"] == right["policy_actions"]
                ),
                "relevant_rule_mode": relevant_rule_mode,
                "relevant_rule_id": relevant_rule,
                "relevant_threshold": threshold,
                "score_wave0": score_wave0,
                "score_wave1": score_wave1,
                "score_absolute_difference": score_difference,
                "threshold_margin_wave0": threshold_margin_wave0,
                "threshold_margin_wave1": threshold_margin_wave1,
                "distance_to_threshold_wave0": abs(threshold_margin_wave0),
                "distance_to_threshold_wave1": abs(threshold_margin_wave1),
                "minimum_distance_to_threshold": minimum_distance,
                "margin_certified": minimum_distance > score_difference,
                "max_score_absolute_difference": max_score_difference,
            }
        )

    additions = [item for event in recorder.events for item in event["added"]]
    removals = [item for event in recorder.events for item in event["removed"]]
    moves = [item for event in recorder.events for item in event["moved"]]
    initialization = [
        check
        for event in recorder.events
        for check in event["initialized_empty"].values()
    ]
    action_count = sum(row["action_record"] is not None for row in outputs)
    output_lengths = sorted({len(row["ordinary_output_tokens"]) for row in outputs})
    summary = {
        "status": "passed",
        "policy": "LMSM-Anytime",
        "max_num_seqs": 32,
        "request_count": len(outputs),
        "harmful_request_count": 32,
        "benign_request_count": 32,
        "action_count": action_count,
        "no_action_count": len(outputs) - action_count,
        "stable_request_ids_exact_and_ordered": output_ids == expected_ids,
        "unique_request_ids": len(set(output_ids)) == len(output_ids),
        "observed_output_lengths": output_lengths,
        "batch_update_event_count": len(recorder.events),
        "addition_count": len(additions),
        "removal_count": len(removals),
        "movement_count": len(moves),
        "slot_reuse_count": sum(item["slot_reused"] for item in additions),
        "all_new_requests_initialized_empty": all(
            all(check.values()) for check in initialization
        ),
        "duplicate_pair_count": len(pair_results),
        "duplicate_pair_decision_exact_count": sum(
            row["decision_exact"] for row in pair_results
        ),
        "duplicate_pair_threshold_crossing_exact_count": sum(
            row["threshold_crossing_exact"] for row in pair_results
        ),
        "duplicate_pair_margin_certified_count": sum(
            row["margin_certified"] for row in pair_results
        ),
        "duplicate_pair_margin_not_certified_count": sum(
            not row["margin_certified"] for row in pair_results
        ),
        "duplicate_pair_margin_not_certified_ids": [
            row["pair_id"] for row in pair_results if not row["margin_certified"]
        ],
        "minimum_duplicate_pair_distance_to_threshold": min(
            row["minimum_distance_to_threshold"] for row in pair_results
        ),
        "max_duplicate_pair_score_absolute_difference": max(
            row["max_score_absolute_difference"] for row in pair_results
        ),
        "pair_results": pair_results,
    }
    passed = (
        summary["stable_request_ids_exact_and_ordered"]
        and summary["unique_request_ids"]
        and len(output_lengths) > 1
        and summary["removal_count"] > 0
        and summary["slot_reuse_count"] > 0
        and summary["all_new_requests_initialized_empty"]
        and summary["duplicate_pair_decision_exact_count"]
        == summary["duplicate_pair_count"]
        and summary["duplicate_pair_threshold_crossing_exact_count"]
        == summary["duplicate_pair_count"]
    )
    summary["status"] = "passed" if passed else "failed"
    return summary


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = prepare_inputs(args.xstest_input)
    input_path = args.run_dir / "inputs.jsonl"
    write_jsonl(input_path, inputs)

    sys.path.insert(0, str(SCRIPTS))
    import run_vllm as runner
    import lmsm.integrations.vllm as runtime

    original_sampling = runner._sampling_params
    caps = {row["id"]: row["request_max_new_tokens"] for row in inputs}

    def varied_sampling(components, run_args, request_id, seed):
        params = original_sampling(components, run_args, request_id, seed)
        params.max_tokens = caps[request_id]
        return params

    recorder_holder = {}
    original_build = runner._build_components

    def recording_build(run_args, profile, project_root):
        components = original_build(run_args, profile, project_root)
        recorder = BatchEventRecorder(runtime.STATE, runtime.REQUEST_ID_EXTRA_ARG)
        runtime.STATE.update_batch = recorder.update
        recorder_holder["recorder"] = recorder
        return components

    runner._sampling_params = varied_sampling
    runner._build_components = recording_build
    run_args = runner.parse_args(
        [
            "--model", args.model,
            "--profile", str(args.profile),
            "--input", str(input_path),
            "--run-dir", str(args.run_dir / "run"),
            "--physical-gpu", str(args.physical_gpu),
            "--sampling", "greedy",
            "--max-new-tokens", "32",
            "--max-model-len", "4096",
            "--max-num-seqs", "32",
            "--max-num-batched-tokens", "32768",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--mode", "eager",
            "--seed", "0",
        ]
    )
    runner.run(run_args)

    recorder = recorder_holder["recorder"]
    summary = analyze(
        inputs,
        read_jsonl(args.run_dir / "run/outputs.jsonl"),
        recorder,
        load_thresholds(args.profile),
    )
    (args.run_dir / "batch_events.json").write_text(
        json.dumps(recorder.events, indent=2) + "\n", encoding="utf-8"
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.run_dir / "duplicate_pair_margins.jsonl", summary["pair_results"])
    print(json.dumps({key: value for key, value in summary.items() if key != "pair_results"}, indent=2))
    if summary["status"] != "passed":
        raise SystemExit("batch-isolation experiment failed")


if __name__ == "__main__":
    main()
