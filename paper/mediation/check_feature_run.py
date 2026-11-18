"""Check one association run of the transcoder or the SAE, written by association.py. Runs on the CPU.

The driver recorded every row the device engine evaluated: the request, its step, the slot the engine updated, the
step's capture check, the feature values the engine computed, and the slot's running maxima after the update.

Association. Each recorded maximum must equal the maximum of the request's previous maxima and its own row's values.
A request's first row starts from the reset state, so another request's evidence in its slot would show here.
A request keeps one slot, no two rows of one forward share a slot, and each step arrives once and in order.
Each admission found a fresh slot, which association.py checks at admission.

Self-consistency. The decision recomputed from the request's own maxima over its consumed steps must match the
recorded decision, step, rule, score and release. Under asynchronous scheduling vLLM may run one more forward for a
finished request and discard its token. Such rows enter the running-maximum check but never the decision.
Usage: python check_feature_run.py RUN_DIR      (writes RUN_DIR/analysis_sf.json)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class FeaturePolicy:
    """The policy as the device engine evaluates it, read from the profile (src/lmsm/integrations/vllm_device.py).
    One backend per active rule, with its features and per-feature thresholds. A rule's signal is the fraction of its
    features whose value (or running maximum, for cumulative_fraction rules) reaches the threshold."""

    def __init__(self, profile_path: Path):
        doc = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        bindings = {b["id"]: b for b in doc["backend_bindings"]}
        library = {r["id"]: r for r in doc["rule_library"]["rules"]}
        rules = [library[rule_id] for rule_id in doc["policy"]["active_rules"]]
        self.backends = []
        for rule in rules:
            config = {**(bindings[rule["binding"]].get("config") or {}), **(rule["condition"].get("config") or {})}
            features = [int(x) for x in config["feature_indices"]]
            base = float(config["feature_threshold"] if "feature_threshold" in config else config["threshold"])
            thresholds = [float(x) for x in config.get("feature_thresholds") or [base] * len(features)]
            self.backends.append({"category": rule["target"], "features": features, "thresholds": thresholds,
                                  "cumulative": config.get("feature_group_rule", "any") == "cumulative_fraction"})
        self.union = sorted({f for b in self.backends for f in b["features"]})
        self.column = {f: i for i, f in enumerate(self.union)}
        self.thresholds = np.full((len(self.backends), len(self.union)), np.inf)
        for b, backend in enumerate(self.backends):
            for feature, threshold in zip(backend["features"], backend["thresholds"]):
                self.thresholds[b, self.column[feature]] = threshold
        self.sizes = np.asarray([max(len(b["features"]), 1) for b in self.backends], dtype=np.float64)
        self.cumulative = np.asarray([b["cumulative"] for b in self.backends])
        self.rule_targets = [r["target"] for r in rules]
        self.rule_ids = [r["id"] for r in rules]
        self.incidence = np.asarray([[float(b["category"] == t) for t in self.rule_targets] for b in self.backends])
        self.members = self.incidence.sum(axis=0)
        self.rule_thresholds = np.asarray([float((r["condition"].get("config") or {}).get("threshold", 1.0))
                                           for r in rules])

    def risk(self, maxima, scores):
        """[rows, rules] risk in float64, with the engine's arithmetic: the mean signal of each rule's backends."""
        cumulative = (maxima[:, None, :] >= self.thresholds[None]).sum(axis=2) / self.sizes[None]
        per_step = (scores[:, None, :] >= self.thresholds[None]).sum(axis=2) / self.sizes[None]
        signal = np.where(self.cumulative[None], cumulative, per_step)
        risk = np.zeros((len(signal), len(self.rule_ids)))
        for r in range(len(self.rule_ids)):
            total = np.zeros(len(signal))
            for b in np.nonzero(self.incidence[:, r] > 0)[0]:
                total = total + signal[:, b] * self.incidence[b, r]
            risk[:, r] = total / max(self.members[r], 1.0)
        return risk


def main(run_dir: Path):
    meta = json.loads((run_dir / "driver_metadata.json").read_text())
    name = meta["args"]["policy"]
    asynchronous = meta["args"]["async_scheduling"] == "True"
    policy = FeaturePolicy(REPO / f"profiles/lmsm_{name}.yaml")
    request_ids = [str(x) for x in meta["request_ids_in_input_order"]]
    outputs = {str(r["request_id"]): r for r in read_jsonl(run_dir / "run/outputs.jsonl")}
    sf = np.load(run_dir / "sf_steps.npz")
    req, step, ok, slot, call = (sf[k].astype(t) for k, t in
                                 (("request_index", np.int64), ("step", np.int64), ("ok", bool), ("slot", np.int64),
                                  ("call", np.int64)))
    scores, maxima = sf["scores"].astype(np.float64), sf["maxima"].astype(np.float64)
    violations = defaultdict(list)
    for rid in meta.get("fresh_state_check_failures") or []:
        violations["fresh_state_failures"].append(rid)
    uncaptured = int((~sf["capture_ok"].astype(bool)).sum())
    uncaptured += (meta.get("instrumentation") or {}).get("selected_feature_instrumentation_problem_count") or 0

    risk = policy.risk(maxima, scores)
    triggered = (risk >= policy.rule_thresholds[None, :]) & ok[:, None]
    exact_scores = bool(np.all(policy.members == 1.0))   # one backend per rule: the sum has one term
    order = np.lexsort((np.arange(len(req)), req))          # by request, then evaluation order
    rows_of = {int(req[c[0]]): c for c in np.split(order, np.flatnonzero(np.diff(req[order])) + 1) if len(c)}
    for (c, s), n in Counter(zip(call.tolist(), slot.tolist())).items():
        if n > 1:
            violations["slot_shared_within_call"].append({"call": c, "slot": s})

    for index, rid in enumerate(request_ids):
        row = outputs[rid]
        tokens, record, execution = row["ordinary_output_tokens"], row.get("action_record"), row["execution_record"]
        chunk = rows_of.get(index, np.zeros(0, dtype=np.int64))
        steps = step[chunk]
        # Association: one slot, steps once and in order, and the running-maximum recursion from the reset state.
        if len(set(slot[chunk].tolist())) > 1:
            violations["slot_changed_within_request"].append(rid)
        if len(steps) != len(set(steps.tolist())):
            violations["duplicate_step"].append(rid)
        if len(steps) > 1 and not (np.diff(steps) > 0).all():
            violations["step_order"].append(rid)
        if len(chunk):
            recorded = maxima[chunk]
            previous = np.vstack([np.zeros((1, recorded.shape[1])), recorded[:-1]])
            expected = np.where(ok[chunk][:, None], np.maximum(previous, scores[chunk]), previous)
            bad = np.flatnonzero((expected != recorded).any(axis=1))
            if len(bad):
                violations["running_max_mismatch"].append({"request": rid, "rows": int(len(bad)),
                                                           "first_step": int(steps[bad[0]])})
        # Rows at a step the request never consumed are discarded forwards, possible only under async scheduling.
        consumed = steps < len(tokens)
        if (~consumed).any() and not asynchronous:
            violations["suffix_row_without_async"].append(rid)
        consumed_rows, consumed_steps = chunk[consumed], steps[consumed].tolist()
        if consumed_steps != list(range(len(tokens))):
            violations["evaluated_steps_not_required_steps"].append(rid)
        # Self-consistency: the first consumed step with a triggered rule, the first such rule in policy order.
        hit = np.flatnonzero(triggered[consumed_rows].any(axis=1))
        expected_decision = None
        if len(hit):
            position = int(consumed_rows[hit[0]])
            r = int(np.flatnonzero(triggered[position])[0])
            expected_decision = {"step": int(step[position]), "target_id": policy.rule_targets[r],
                                 "rule_id": policy.rule_ids[r], "score": float(risk[position, r]),
                                 "threshold": float(policy.rule_thresholds[r])}
        if record is None:
            if expected_decision is not None:
                violations["crossing_without_action"].append(rid)
            if execution.get("decision") != "allow" or execution.get("decision_step") is not None:
                violations["record_decision_mismatch"].append(rid)
        else:
            if str(record.get("request_id")) != rid:
                violations["action_record_for_other_request"].append(rid)
            if expected_decision is None:
                violations["action_without_own_crossing"].append(rid)
            elif record.get("step") != expected_decision["step"]:
                violations["decision_step_mismatch"].append(rid)
            elif (record.get("target_id"), record.get("rule_id")) != (expected_decision["target_id"],
                                                                      expected_decision["rule_id"]):
                violations["decision_rule_mismatch"].append(rid)
            else:
                # Written as "not (difference <= tolerance)", so a missing or NaN score also counts as a mismatch.
                score, tolerance = record.get("score"), 0.0 if exact_scores else 1e-12
                if not (isinstance(score, (int, float)) and abs(score - expected_decision["score"]) <= tolerance):
                    violations["decision_score_mismatch"].append(rid)
                if record.get("threshold") != expected_decision["threshold"]:
                    violations["action_threshold_mismatch"].append(rid)
            if len(tokens) != (record.get("step") or 0) + 1 or tokens[-1] != record.get("forced_termination_token"):
                violations["forced_eos_not_at_own_step"].append(rid)
            if row.get("finish_reason") != "stop":
                violations["acted_but_not_stopped"].append(rid)
            if execution.get("decision") != record.get("action") or execution.get("decision_step") != record.get("step"):
                violations["record_decision_mismatch"].append(rid)
        if execution.get("generated_token_count") != len(tokens) or execution.get("processed_positions") != len(tokens):
            violations["record_token_count_mismatch"].append(rid)
        # The archived feature maxima must equal the recorded state after the last consumed row (or a later row).
        if len(chunk):
            archived = row.get("feature_maxima") or {}
            vector = np.full(len(policy.union), np.nan)
            for backend in policy.backends:
                values = archived.get(backend["category"]) or {}
                for feature in backend["features"]:
                    value = values.get(str(feature))
                    if value is not None:
                        vector[policy.column[feature]] = float(value)
            last = consumed_steps[-1] if consumed_steps else 0
            if not any(np.array_equal(maxima[p], vector) for p in chunk[steps >= last]):
                violations["archived_maxima_mismatch"].append(rid)

    association = ("running_max_mismatch", "slot_changed_within_request", "slot_shared_within_call",
                   "duplicate_step", "step_order", "fresh_state_failures")
    out = {
        "policy": name,
        "association": {k: len(violations[k]) for k in association},
        "self_consistency": {k: len(v) for k, v in violations.items() if k not in association},
        "uncaptured_rows": uncaptured,
        "violation_examples": {k: v[:10] for k, v in violations.items() if v},
        "execution": {"states": dict(Counter(r["execution_record"]["state"] for r in outputs.values())),
                      "fresh_state_checks": meta.get("fresh_state_checks"),
                      "fresh_state_check_failures": len(meta.get("fresh_state_check_failures") or [])},
    }
    clean = not any(out["association"].values()) and not any(out["self_consistency"].values()) and not uncaptured
    out["status"] = "VALID" if clean else "VIOLATION"
    (run_dir / "analysis_sf.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run_dir), "status": out["status"], "association": out["association"],
                      "self_consistency": out["self_consistency"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
