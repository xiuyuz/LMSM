#!/usr/bin/env python3
"""The numbers of Section 6.3, Tables 5 and 6, and the appendix paragraph on generation and judging.

Reads the runs under $RUNS/effectiveness (README.md lists the commands) and writes results.json. Each rate is a
percentage per seed, summarized by its mean over the seeds and its lowest and highest seed. An unknown judge label
stops the analysis, except in Table 6, which leaves out the responses that WildGuard did not label.

Usage: analyze.py [OUT.json]   (default: results.json next to this script)
"""

import csv
import functools
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "judges"))
import labels  # noqa: E402

RUNS = Path(os.environ.get("RUNS", HERE.parent / "runs")) / "effectiveness"
SEEDS = (0, 1, 2)
ARMS = ("disabled", "checkpoint", "anytime")
JUDGE = {"harmbench": labels.unsafe, "wildjailbreak": labels.unsafe, "strongreject": labels.unsafe,
         "xstest": labels.refusal}
# Qwen3-4B without safety training and with six training-time defenses, from Table 1 of ThinkSafe (lee2026thinksafe):
# HarmBench, StrongREJECT and WildJailbreak ASR, XSTest FRR and their mean, the safety average.
THINKSAFE = {"No safety training": (38.21, 8.31, 43.00, 0.80, 22.58), "DirectRefusal": (33.06, 3.19, 36.20, 32.00, 26.11),
             "SafeChain": (43.69, 41.21, 39.65, 2.00, 31.64), "STAR-1": (33.72, 5.75, 35.15, 6.80, 20.36),
             "SafePath": (37.71, 7.35, 42.45, 1.60, 22.28), "SafeKey": (32.39, 3.19, 32.95, 0.80, 17.33),
             "ThinkSafe": (9.63, 0.32, 7.45, 2.80, 5.05)}


# ---- reading the runs ---------------------------------------------------------------------------------------------
def outputs(setting, bench, seed, arm):
    with open(RUNS / setting / bench / f"seed_{seed}" / arm / "outputs.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@functools.lru_cache(maxsize=None)
def scored(setting, bench, seed, judge="judge"):
    return labels.read_scored(RUNS / setting / bench / f"seed_{seed}" / judge / "scored.jsonl")


@functools.lru_cache(maxsize=None)
def new_pairs():
    return labels.read_scored(RUNS / "filter" / "new_pairs" / "judge" / "scored.jsonl")["new_pairs"]


def timings(round_, arm):
    with open(RUNS / "filter_timing" / f"round{round_}" / arm / "timings.csv", newline="", encoding="utf-8") as f:
        return [float(r["generate_seconds"]) for r in csv.DictReader(f) if r["is_warmup"] == "False"]


def workload(name):
    with open(RUNS / "filter_timing" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- counting -----------------------------------------------------------------------------------------------------
def summary(per_seed):
    return {"mean": statistics.mean(per_seed), "min": min(per_seed), "max": max(per_seed), "per_seed": per_seed}


def rate(values):
    """The share of "yes" among the labels, which must all be known."""
    if "unknown" in values:
        raise SystemExit(f"{values.count('unknown')} unknown judge labels")
    return 100 * values.count("yes") / len(values)


def known_rate(values):
    """The share of "yes" among the known labels, and the number of unknown labels."""
    unknown = values.count("unknown")
    return 100 * values.count("yes") / (len(values) - unknown), unknown


def judged(setting, bench, arm, keep=None):
    return [rate([JUDGE[bench](r) for rid, r in scored(setting, bench, s)[arm].items() if keep is None or rid in keep])
            for s in SEEDS]


# ---- Table 5 and the probe policies -------------------------------------------------------------------------------
def table5():
    table = {}
    for arm in ARMS:
        per = {b: judged("matched", b, arm) for b in ("harmbench", "wildjailbreak", "xstest")}
        per["strongreject"] = judged("defaults", "strongreject", arm)  # StrongREJECT ran with vLLM's defaults only
        per["mean"] = [statistics.mean(per[b][i] for b in ("harmbench", "strongreject", "wildjailbreak", "xstest"))
                       for i in range(len(SEEDS))]
        table[arm] = {b: summary(v) for b, v in per.items()}
    return table


def intervention_steps():
    """The generated-token step of each refusal or termination under matched settings, HarmBench and WildJailbreak."""
    steps = {}
    for arm in ("checkpoint", "anytime"):
        steps[arm] = [r["action_record"]["step"] for b in ("harmbench", "wildjailbreak") for s in SEEDS
                      for r in outputs("matched", b, s, arm)
                      if r.get("action_record") and r["action_record"]["action"] in ("refuse", "terminate")]
    return {arm: {"median": statistics.median(v), "interventions": len(v)} for arm, v in steps.items()}


# ---- further checks -----------------------------------------------------------------------------------------------
def never_seen():
    """HarmBench prompts whose behavior never appeared during development, with vLLM's defaults."""
    development = set(json.loads((HERE.parent / "data" / "splits.json").read_text())["development_behaviors"])
    objective = {str(r["request_id"]): r["objective"] for r in outputs("defaults", "harmbench", 0, "disabled")}
    keep = {rid for rid, o in objective.items() if o not in development}
    return {"prompts": len(keep), "asr": {arm: summary(judged("defaults", "harmbench", arm, keep))
                                           for arm in ("disabled", "checkpoint")}}


def harmbench_classifier():
    out = {}
    for arm in ("disabled", "checkpoint"):
        rows = [scored("defaults", "harmbench", s, "harmbench_classifier")[arm] for s in SEEDS]
        out[arm] = summary([rate([labels.harmbench_yes(r) for r in seed_rows.values()]) for seed_rows in rows])
    out["judged_requests"] = len(rows[0])
    return out


def ordinary():
    per_seed, by_dataset, n = [], Counter(), 0
    for s in SEEDS:
        rows = outputs("defaults", "ordinary", s, "checkpoint")
        acted = [r for r in rows if r["execution_record"]["decision"] in ("refuse", "terminate")]
        per_seed.append(100 * len(acted) / len(rows))
        by_dataset.update(r["dataset"] for r in acted)
        n = len(rows)
    return {"requests": n, "checkpoint_intervenes": summary(per_seed),
            "interventions_by_dataset_three_seeds": dict(by_dataset.most_common())}


# ---- Table 6 ------------------------------------------------------------------------------------------------------
def table6():
    """WildGuard labels of the filter, Checkpoint and both together, without the responses it left unlabeled."""
    field = {"harmbench": labels.harmful_response, "wildjailbreak": labels.harmful_response,
             "xstest": lambda r: labels.refusal(r, hand_read=False)}
    cells = {arm: {} for arm in ("filter", "checkpoint", "combined")}
    unlabeled = []  # the share of unlabeled responses of each cell, over its three seeds
    for bench in ("harmbench", "wildjailbreak", "xstest"):
        per = {arm: [] for arm in cells}
        unknown = {arm: [0, 0] for arm in cells}
        for s in SEEDS:
            judge = scored("filter", bench, s)
            # Each combined response takes the label of the same pair, from Checkpoint first, then Anytime and the filter.
            donors = {(rid, r["response"]): r for arm in ("filtered", "anytime", "checkpoint") for rid, r in judge[arm].items()}
            combined = [donors.get((str(r["request_id"]), r["released_output"]))
                        or new_pairs()[f"{bench}/seed_{s}/{r['request_id']}"]
                        for r in outputs("filter", bench, s, "combined")]
            for arm, rows in (("filter", list(judge["filtered"].values())), ("checkpoint", list(judge["checkpoint"].values())),
                              ("combined", combined)):
                value, missing = known_rate([field[bench](r) for r in rows])
                per[arm].append(value)
                unknown[arm][0] += missing
                unknown[arm][1] += len(rows)
        for arm in cells:
            cells[arm][bench] = summary(per[arm])
            unlabeled.append(100 * unknown[arm][0] / unknown[arm][1])

    median = {arm: statistics.median(timings(1, arm) + timings(2, arm))
              for arm in ("native", "checkpoint", "filter", "filter_after")}
    cells["filter"]["time_added"] = 100 * median["filter"] / median["native"]
    cells["checkpoint"]["time_added"] = 100 * (median["checkpoint"] - median["native"]) / median["native"]
    cells["combined"]["time_added"] = 100 * (median["checkpoint"] - median["native"] + median["filter_after"]) / median["native"]
    released = workload("workload_checkpoint.jsonl")
    return cells, {"largest_unlabeled_share": max(unlabeled), "new_pairs": len(new_pairs()),
                   "timing_requests": len(released), "timing_requests_per_source": dict(Counter(r["source"] for r in released)),
                   "timing_fixed_refusals": sum(r["checkpoint_refuses"] for r in released),
                   "timing_median_seconds": median}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    t5 = table5()
    defaults = {arm: {b: summary(judged("defaults", b, arm)) for b in ("harmbench", "wildjailbreak", "xstest")}
                for arm in ARMS}
    t6, filter_details = table6()
    reported = {k: dict(zip(("harmbench", "strongreject", "wildjailbreak", "xstest", "mean"), v)) for k, v in THINKSAFE.items()}
    lmsm_cells = [t5[a][b] for a in ("checkpoint", "anytime") for b in ("harmbench", "strongreject", "wildjailbreak", "xstest")]
    results = {
        "table5": t5,
        "table5_reported_by_thinksafe": reported,
        "table6": t6,
        "vllm_defaults": defaults,
        "intervention_steps_matched": intervention_steps(),
        "never_seen": never_seen(),
        "harmbench_classifier": harmbench_classifier(),
        "ordinary_requests": ordinary(),
        "table6_details": filter_details,
        "text": {
            "wildjailbreak_reduction_factor": t5["disabled"]["wildjailbreak"]["mean"] / t5["checkpoint"]["wildjailbreak"]["mean"],
            "largest_seed_range_of_an_lmsm_rate": max(c["max"] - c["min"] for c in lmsm_cells),
            "largest_change_with_vllm_defaults": max(abs(defaults[a][b]["mean"] - t5[a][b]["mean"])
                                                     for a in ARMS for b in ("harmbench", "wildjailbreak", "xstest")),
            "largest_gap_no_safety_training_to_disabled": max(abs(reported["No safety training"][b] - t5["disabled"][b]["mean"])
                                                              for b in ("harmbench", "strongreject", "wildjailbreak", "xstest")),
            "lowest_reported_mean": min(v["mean"] for k, v in reported.items() if k != "No safety training"),
            "checkpoint_over_filter_harmful_releases": {b: t6["checkpoint"][b]["mean"] / t6["filter"][b]["mean"]
                                                        for b in ("harmbench", "wildjailbreak")},
            "checkpoint_over_filter_time": t6["checkpoint"]["time_added"] / t6["filter"]["time_added"],
            "checkpoint_minus_filter_frr": t6["checkpoint"]["xstest"]["mean"] - t6["filter"]["xstest"]["mean"],
            "combined_over_checkpoint_harmful_releases": {b: t6["combined"][b]["mean"] / t6["checkpoint"][b]["mean"]
                                                          for b in ("harmbench", "wildjailbreak")},
            "combined_minus_checkpoint_frr": t6["combined"]["xstest"]["mean"] - t6["checkpoint"]["xstest"]["mean"],
        },
    }
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
