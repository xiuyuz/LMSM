"""Check one association run of a probe policy, written by association.py. Runs on the CPU.

Self-consistency: each request's decision must follow from its own recorded evidence. A refusal must sit at the
request's own first crossing, with its own score, on its own output. An allowed request must cross no rule, and the
execution record must agree. The evaluated steps must be exactly the steps the policy requires.
Fresh state: each admitted request must start from empty policy state, which association.py checks at admission.
Usage: python check_probe_run.py RUN_DIR      (writes RUN_DIR/summary.json)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(run_dir: Path):
    meta = json.loads((run_dir / "driver_metadata.json").read_text())
    policy = meta["args"]["policy"]
    parameters = json.loads((REPO / f"profiles/parameters/lmsm_{policy}.json").read_text())
    thresholds = {rule: float(v["threshold"])
                  for rule, v in parameters["rules" if policy == "checkpoint" else "probes"].items()}
    active_order = yaml.safe_load((REPO / f"profiles/lmsm_{policy}.yaml").read_text())["policy"]["active_rules"]
    request_ids = meta["request_ids_in_input_order"]
    index_of = {rid: i for i, rid in enumerate(request_ids)}
    outputs = {row["request_id"]: row for row in read_jsonl(run_dir / "run/outputs.jsonl")}

    # The steps at which the policy received each request's evidence.
    evidence = np.load(run_dir / "evidence.npz")
    evaluated = defaultdict(list)
    for index, step in zip(evidence["request_index"].tolist(), evidence["step"].tolist()):
        evaluated[index].append(step)

    # Anytime: every per-step score of the 15 probes, in step order per request.
    step_scores, probe_order = {}, []
    if policy == "anytime":
        data = np.load(run_dir / "anytime_step_scores.npz")
        probe_order = [str(name) for name in data["probe_order"]]
        scores = data["scores"]
        rows_of = defaultdict(list)
        for position, (index, step) in enumerate(zip(data["request_index"].tolist(), data["step"].tolist())):
            rows_of[index].append((step, position))
        for index, items in rows_of.items():
            items.sort()
            step_scores[index] = (np.asarray([s for s, _ in items]), scores[[p for _, p in items]])
        threshold_vector = np.asarray([thresholds[name] for name in probe_order])

    violations = defaultdict(list)
    for rid in request_ids:
        row, index = outputs[rid], index_of[rid]
        tokens, record, execution = row["ordinary_output_tokens"], row["action_record"], row["execution_record"]
        if policy == "anytime":
            # The decision rests on the running maximum of each probe's per-step score.
            steps, values = step_scores.get(index, (np.asarray([], dtype=int), None))
            expected = None
            if len(steps):
                running = np.maximum.accumulate(values, axis=0)
                crossed = running >= threshold_vector
                hits = np.nonzero(crossed.any(axis=1))[0]
                if len(hits):
                    first = int(hits[0])
                    rules = [probe_order[k] for k in np.nonzero(crossed[first])[0]]
                    rule = next(r for r in active_order if r in rules)
                    expected = {"step": int(steps[first]), "target_id": rule,
                                "score": float(running[first, probe_order.index(rule)])}
            if steps.tolist() != list(range(1, (record["step"] if record else len(tokens) - 1) + 1)):
                violations["evaluated_steps_not_required_steps"].append(rid)
        else:
            # Checkpoint decides once, on the fusion scores at token 64.
            fusion = row["policy_scores"].get("fusion", {})
            rules = [r for r in active_order if r in fusion and fusion[r] >= thresholds[r]]
            expected = {"step": 64, "target_id": rules[0], "score": float(fusion[rules[0]])} if rules else None
            required = list(range(65)) if len(tokens) - 1 >= 64 else list(range(len(tokens)))
            if sorted(evaluated.get(index, [])) != required:
                violations["evaluated_steps_not_required_steps"].append(rid)
        if record is None:
            if expected is not None:
                violations["crossing_without_action"].append(rid)
            if any(row["policy_actions"].values()):
                violations["crossing_flag_without_action"].append(rid)
            if execution["decision"] != "allow" or execution["decision_step"] is not None:
                violations["record_decision_mismatch"].append(rid)
        else:
            if record["request_id"] != rid:
                violations["action_record_for_other_request"].append(rid)
            if expected is None:
                violations["action_without_own_crossing"].append(rid)
            elif (record["step"], record["target_id"]) != (expected["step"], expected["target_id"]):
                violations["action_step_or_rule_not_own_first_crossing"].append(rid)
            elif record["score"] != expected["score"]:
                violations["action_score_not_own_score"].append(rid)
            if len(tokens) != record["step"] + 1 or tokens[-1] != record["forced_termination_token"]:
                violations["forced_eos_not_at_own_step"].append(rid)
            if row["finish_reason"] != "stop":
                violations["acted_but_not_stopped"].append(rid)
            if not row["policy_actions"].get(record["target_id"]):
                violations["action_rule_flag_false"].append(rid)
            if execution["decision"] != record["action"] or execution["decision_step"] != record["step"]:
                violations["record_decision_mismatch"].append(rid)
        if execution["generated_token_count"] != len(tokens) or execution["processed_positions"] != len(tokens):
            violations["record_token_count_mismatch"].append(rid)

    additions = [item for event in json.loads((run_dir / "batch_events.json").read_text()) for item in event["added"]]
    summary = {
        "policy": policy,
        "request_count": len(outputs),
        "self_consistency": {
            "requests_checked": len(request_ids),
            "violations": {key: len(value) for key, value in violations.items()},
            "violation_examples": {key: value[:5] for key, value in violations.items()},
            "requests_with_any_violation": len({rid for value in violations.values() for rid in value}),
        },
        "batching": {
            "slot_reuses": sum(item["slot_reused"] for item in additions),
            "fresh_state_checks": meta.get("fresh_state_checks"),
            "fresh_state_check_failures": len(meta.get("fresh_state_check_failures") or []),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run_dir), "self_consistency_violations": summary["self_consistency"]
                      ["requests_with_any_violation"], **summary["batching"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
