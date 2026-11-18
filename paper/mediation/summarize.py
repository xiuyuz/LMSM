"""Turn the runs under $RUNS/mediation into results.json, the numbers of Table 4, Figure 5 and Section 6.2.

It reads the per-run checks of the association runs (summary.json and analysis_sf.json), compares each ablation run
with its request-keyed control request by request, and counts the outcomes of the contract tests, the short answers
and the released-artifact policies of $RUNS/released_artifacts.
Usage: python summarize.py [RUNS]
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUNS", HERE.parent / "runs"))
M = RUNS / "mediation"
REFUSED_AT_STARTUP = "LMSM refuses to start"


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def decision(row):
    rec = row.get("action_record")
    return (rec["action"], rec["rule_id"], rec["target_id"], rec["step"]) if rec else (None, None, None, None)


def refused_with_fixed_text(row, refusal):
    rec = row["execution_record"]
    return rec["state"] == "execution_failed" and rec["release_outcome"] == "refusal" and row["released_output"] == refusal


def association():
    probes = [json.loads(p.read_text()) for p in sorted(M.glob("association/probes/*/summary.json"))]
    meta = [json.loads(p.read_text()) for p in sorted(M.glob("association/probes/*/driver_metadata.json"))]
    features = [json.loads(p.read_text()) for p in sorted(M.glob("association/*/*/analysis_sf.json"))]
    p = {"runs": len(probes), "requests": sum(s["request_count"] for s in probes),
         "association_errors": sum(s["self_consistency"]["requests_with_any_violation"] for s in probes)
         + sum(s["batching"]["fresh_state_check_failures"] for s in probes),
         "fewest_slot_reuses": min(s["batching"]["slot_reuses"] for s in probes),
         "runs_with_chunked_prefill_and_prefix_caching": sum(
             m["args"].get("enable_chunked_prefill") == "True" and m["args"].get("enable_prefix_caching") == "True"
             for m in meta)}
    f = {"runs": len(features), "valid_runs": sum(a["status"] == "VALID" for a in features),
         "requests": sum(sum(a["execution"]["states"].values()) for a in features),
         "association_errors": sum(sum(a["association"].values()) + sum(a["self_consistency"].values())
                                   + a["execution"]["fresh_state_check_failures"] for a in features)}
    return {"probes": p, "released_artifacts": f, "requests": p["requests"] + f["requests"]}


def pairs(run_dir):
    """The two copies of each prompt in one run: how many pairs decided identically, and how many in total."""
    by = {r["request_id"]: r for r in rows(run_dir / "run/outputs.jsonl")}
    ids = sorted({rid.rsplit("__c", 1)[0] for rid in by})
    same = sum(decision(by[f"{i}__c0"]) == decision(by[f"{i}__c1"]) for i in ids)
    exact = sum(decision(by[f"{i}__c0"]) == decision(by[f"{i}__c1"])
                and (by[f"{i}__c0"].get("policy_actions") or {}) == (by[f"{i}__c1"].get("policy_actions") or {})
                and by[f"{i}__c0"]["released_output_tokens"] == by[f"{i}__c1"]["released_output_tokens"]
                and by[f"{i}__c0"]["ordinary_output_tokens"] == by[f"{i}__c1"]["ordinary_output_tokens"] for i in ids)
    return same, exact, len(ids)


def duplicate_pairs():
    bi = [pairs(d) for d in sorted(M.glob("ablation/bi/*_control_*")) if d.is_dir()]
    default = [pairs(d) for d in sorted(M.glob("ablation/default/*_control_*")) if d.is_dir()]
    differing = [100 * (n - same) / n for same, _, n in default]
    return {"batch_invariant": {"runs": len(bi), "pairs": sum(n for *_, n in bi),
                                "identical": sum(exact for _, exact, _ in bi)},
            "default_kernels": {"runs": len(default), "differing_pct": [min(differing), max(differing)]}}


def compare(control, ablated):
    """An ablation run against its request-keyed control, request by request."""
    ctrl = {r["request_id"]: r for r in rows(control / "run/outputs.jsonl")}
    out = {"requests": 0, "used_another_requests_state": 0, "failed_closed": 0, "decisions_changed": 0,
           "harmbench_refusals_released": 0}
    for r in rows(ablated / "run/outputs.jsonl"):
        rec = r["execution_record"]
        failed = rec["state"] == "execution_failed"
        out["requests"] += 1
        out["failed_closed"] += failed
        out["used_another_requests_state"] += not failed and bool(
            (rec.get("association_ablation") or {}).get("decision_used_foreign_evidence"))
        if not failed:
            before, after = decision(ctrl[r["request_id"]]), decision(r)
            out["decisions_changed"] += before != after
            out["harmbench_refusals_released"] += r["source"] == "harmbench" and before[0] == "refuse" and after[0] is None
    return out


def ablation():
    """Figure 5: batch-invariant kernels. Anytime tiles count decisions on another request's state, Checkpoint tiles
    count requests that failed closed. The controls must complete every request."""
    root = M / "ablation/bi"
    cells = {}
    for policy in ("anytime", "checkpoint"):
        for arrival in ("burst", "staggered"):
            control = root / f"{policy}_control_{arrival}"
            states = {r["execution_record"]["state"] for r in rows(control / "run/outputs.jsonl")}
            cells[f"{policy}/control/{arrival}"] = {"control_states": sorted(states)}
            for cond in ("rowkey_reset", "rowkey_noreset"):
                cells[f"{policy}/{cond}/{arrival}"] = compare(control, root / f"{policy}_{cond}_{arrival}")
    return cells


def contract():
    out = {"refused_at_startup": 0, "refused_runs_without_outputs": 0, "capture_removed": 0,
           "capture_removed_failed_closed": 0, "observation_dropped": 0, "observation_dropped_failed_closed": 0,
           "other_requests_completed": True, "device_vs_host": 0, "device_vs_host_same": 0}
    for suite in ("matched", "defaults"):
        s = M / "contract" / suite
        out["refused_at_startup"] += sum(REFUSED_AT_STARTUP in p.read_text(errors="replace")
                                         for p in s.glob("logs/startup_*.log"))
        out["refused_runs_without_outputs"] += sum(not (d / "outputs.jsonl").exists() for d in s.glob("startup/*"))
        for path in s.glob("capture_removed/*/*/outputs.jsonl"):
            refusal = json.loads((path.parent / "metadata.json").read_text())["refusal_text"]
            got = rows(path)
            out["capture_removed"] += len(got)
            out["capture_removed_failed_closed"] += sum(refused_with_fixed_text(r, refusal) for r in got)
        for policy in ("checkpoint", "anytime"):
            run = s / "dropped_observation" / policy / "run"
            refusal = json.loads((run / "metadata.json").read_text())["refusal_text"]
            targets = {t["request_id"] for t in json.loads((run.parent / "targets.json").read_text())["targets"]}
            for r in rows(run / "outputs.jsonl"):
                if r["request_id"] in targets:
                    out["observation_dropped"] += 1
                    out["observation_dropped_failed_closed"] += refused_with_fixed_text(r, refusal)
                elif r["execution_record"]["state"] != "completed":
                    out["other_requests_completed"] = False
            host, device = (s / "device_vs_host" / policy / ev / "outputs.jsonl" for ev in ("host", "device"))
            if host.exists():
                h = {r["request_id"]: r for r in rows(host)}
                for d in rows(device):
                    x = h[d["request_id"]]
                    out["device_vs_host"] += 1
                    out["device_vs_host_same"] += (decision(x) == decision(d)
                                                   and x["released_output_tokens"] == d["released_output_tokens"]
                                                   and x["execution_record"]["state"] == d["execution_record"]["state"])
    out["failed_requests"] = out["capture_removed"] + out["observation_dropped"]
    out["failed_requests_fixed_refusal"] = out["capture_removed_failed_closed"] + out["observation_dropped_failed_closed"]
    return out


def not_reached():
    """A request is not reached exactly when its answer ends by token 64, and its release follows the declared
    treatment: an ordinary release with a record, or the fixed refusal."""
    out = {"requests": 0, "mismatches": 0}
    for path in sorted(M.glob("not_reached/*/lmsm_checkpoint*/outputs.jsonl")):
        deny = path.parent.name.endswith("_deny")
        refusal = json.loads((path.parent / "metadata.json").read_text())["refusal_text"]
        for r in rows(path):
            rec = r["execution_record"]
            missed = rec["state"] == "not_reached"
            ok = rec["state"] == "execution_failed" or missed == (len(r["ordinary_output_tokens"]) <= 64)
            if missed:
                ok &= (rec["release_outcome"] == "refusal" and r["released_output"] == refusal if deny else
                       rec["release_outcome"] == "ordinary"
                       and r["released_output_tokens"] == r["ordinary_output_tokens"])
            out["requests"] += 1
            out["mismatches"] += not ok
    return out


def released_as_decided():
    """Every request of the deployed transcoder and SAE policies (all six rules, enforcing) completed its account and
    was released as its decision required."""
    out = {"requests": 0, "mismatches": 0}
    found = [Path(d) / "outputs.jsonl" for d, _, files in os.walk(RUNS / "released_artifacts", followlinks=True)
             if "outputs.jsonl" in files]
    for path in sorted(found):
        meta_path = path.parent / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        profile = meta.get("profile") or {}
        if (meta.get("observe_only") or profile.get("name") not in ("LMSM-Transcoder", "LMSM-SAE")
                or len(profile["policy"]["active_rule_ids"]) != len(profile["rule_library"]["rules"])):
            continue
        for r in rows(path):
            rec = r["execution_record"]
            refused = rec["decision"] in ("refuse", "terminate")
            ok = rec["state"] == "completed" and (
                rec["release_outcome"] == "refusal" and r["released_output"] == meta["refusal_text"] if refused else
                rec["release_outcome"] == "ordinary" and r["released_output_tokens"] == r["ordinary_output_tokens"])
            out["requests"] += 1
            out["mismatches"] += not ok
    return out


def main():
    res = {"association": association(), "duplicate_pairs": duplicate_pairs(), "ablation": ablation(),
           "contract": contract(), "not_reached": not_reached(), "released_as_decided": released_as_decided()}
    (HERE / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
