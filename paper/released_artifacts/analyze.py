#!/usr/bin/env python3
"""The numbers of Section 6.4, Figure 6, and the appendix paragraphs on released-artifact rules and rule replay.

A rule fires when the fraction of its features whose running maximum reaches the feature's threshold is at least
the rule's required fraction. The maxima only grow and the gate holds each response until generation ends, so the
final maxima of a Disabled run decide what any subset of rules would have refused. The replay counts a harmful
release when the judge calls a response unsafe and no active rule fires, and a false refusal when the judge calls
it a refusal or an active rule fires.

Usage: analyze.py [OUT.json]   (default: results.json next to this script)
"""

import functools
import json
import os
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "judges"))
import labels  # noqa: E402

RUNS = Path(os.environ.get("RUNS", HERE.parent / "runs")) / "released_artifacts"
PROFILES = HERE.parents[1] / "profiles"
SPLITS = json.loads((HERE.parent / "data" / "splits.json").read_text())
CATEGORY = SPLITS["harmbench_categories"]
SEEDS = (0, 1, 2)
RULES = ("chemical_biological", "cybercrime_intrusion", "illegal_activity", "misinformation_disinformation",
         "book_copyright", "lyrics_copyright")
TARGET = {"chemical_biological": "chemical_biological", "cybercrime_intrusion": "cybercrime_intrusion",
          "illegal_activity": "illegal", "misinformation_disinformation": "misinformation_disinformation"}
COLUMNS = ("chemical_biological", "cybercrime_intrusion", "illegal", "misinformation_disinformation",
           "harassment_bullying", "harmful", "xstest")
BACKENDS = {
    # The transcoder replays on the combined policy's Disabled run (Qwen3-4B, matched settings), which records the
    # transcoder features under the prefix "transcoder/". The SAE replays on its own Disabled run (Gemma-3-4B,
    # vLLM's default optimizations).
    "transcoder": {"profile": "lmsm_transcoder.yaml", "replay": ("combined", "transcoder/"),
                   "deployed": ("transcoder_matched", "transcoder"), "checked": ("transcoder_defaults", "transcoder")},
    "sae": {"profile": "lmsm_sae.yaml", "replay": ("sae_defaults", ""),
            "deployed": ("sae_defaults", "sae"), "checked": ("sae_defaults", "sae")},
}


# ---- reading the runs ---------------------------------------------------------------------------------------------
def outputs(experiment, bench, seed, arm):
    with open(RUNS / experiment / bench / f"seed_{seed}" / arm / "outputs.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@functools.lru_cache(maxsize=None)
def scored(experiment, bench, seed):
    return labels.read_scored(RUNS / experiment / bench / f"seed_{seed}" / "judge" / "scored.jsonl")


def copyright_classified(backend, seed):
    path = RUNS / "copyright" / backend / "requests" / f"seed_{seed}" / "disabled" / "copyright.jsonl"
    with open(path, encoding="utf-8") as f:
        return {r["request_id"]: r["reproduces"] for r in map(json.loads, f)}


# ---- rules --------------------------------------------------------------------------------------------------------
def rules(profile):
    """Each rule's feature indices, feature thresholds and required fraction, from the profile."""
    config = {r["id"]: r["condition"]["config"]
              for r in yaml.safe_load((PROFILES / profile).read_text())["rule_library"]["rules"]}
    return {r: ([str(i) for i in config[r]["feature_indices"]], config[r]["feature_thresholds"], config[r]["threshold"])
            for r in RULES}


def fires(rule_set, maxima, prefix=""):
    """The rules whose features reach their thresholds in the final running maxima."""
    out = set()
    for rule, (indices, thresholds, fraction) in rule_set.items():
        values = maxima[prefix + rule]
        if sum(values.get(i, 0.0) >= t for i, t in zip(indices, thresholds)) / len(indices) >= fraction:
            out.add(rule)
    return out


def summary(per_seed):
    return {"mean": statistics.mean(per_seed), "min": min(per_seed), "max": max(per_seed), "per_seed": per_seed}


def yes(label):
    if label == "unknown":
        raise SystemExit("an unknown judge label")
    return label == "yes"


# ---- replay on the Disabled runs ----------------------------------------------------------------------------------
def disabled_requests(backend):
    """Per seed, each HarmBench and XSTest request of the Disabled run: the rules that fire, its label and split."""
    experiment, prefix = BACKENDS[backend]["replay"]
    rule_set = rules(BACKENDS[backend]["profile"])
    calibration = set(SPLITS["artifact_calibration_behaviors"])
    held_xstest = set(SPLITS["artifact_held_out_xstest_ids"])
    seeds = []
    for s in SEEDS:
        hb_labels, xs_labels = scored(experiment, "harmbench", s)["disabled"], scored(experiment, "xstest", s)["disabled"]
        hb = [{"fires": fires(rule_set, r["feature_maxima"], prefix), "unsafe": yes(labels.unsafe(hb_labels[str(r["request_id"])])),
               "category": CATEGORY.get(r["objective"]), "held_out": r["objective"] not in calibration}
              for r in outputs(experiment, "harmbench", s, "disabled")]
        xs = [{"fires": fires(rule_set, r["feature_maxima"], prefix), "refusal": yes(labels.refusal(xs_labels[str(r["request_id"])])),
               "held_out": int(r["source_id"]) in held_xstest}
              for r in outputs(experiment, "xstest", s, "disabled")]
        seeds.append((hb, xs))
    return seeds


def replay(seeds, active, held_out=False):
    """HarmBench ASR overall and on each category, and XSTest FRR, per seed, with only the active rules on."""
    out = {"asr": [], "frr": [], **{f"asr_{t}": [] for t in TARGET.values()}}
    for hb, xs in seeds:
        hb = [r for r in hb if r["held_out"] or not held_out]
        xs = [r for r in xs if r["held_out"] or not held_out]

        def released(rows):
            return 100 * sum(r["unsafe"] and not r["fires"] & active for r in rows) / len(rows)

        out["asr"].append(released(hb))
        out["frr"].append(100 * sum(r["refusal"] or bool(r["fires"] & active) for r in xs) / len(xs))
        for t in TARGET.values():
            out[f"asr_{t}"].append(released([r for r in hb if r["category"] == t]))
    return {k: summary(v) for k, v in out.items()}


def firing(seeds, held_out=False):
    """Figure 6: the share of prompts in each column on which each category rule alone fires."""
    cells = {}
    for rule in TARGET:
        cells[rule] = {}
        for col in COLUMNS:
            per = []
            for hb, xs in seeds:
                rows = xs if col == "xstest" else [r for r in hb if r["category"] == col]
                rows = [r for r in rows if r["held_out"] or not held_out]
                per.append(100 * sum(rule in r["fires"] for r in rows) / len(rows))
            cells[rule][col] = summary(per)
    return cells


# ---- the deployed policies ----------------------------------------------------------------------------------------
def agreement(backend):
    """A deployed request is refused exactly when a rule fires, and its record names a rule that fires."""
    experiment, arm = BACKENDS[backend]["checked"]
    rule_set, counts = rules(BACKENDS[backend]["profile"]), {"requests": 0, "agree": 0}
    for bench in ("harmbench", "xstest"):
        for s in SEEDS:
            for r in outputs(experiment, bench, s, arm):
                fired, action = fires(rule_set, r["feature_maxima"]), r.get("action_record") or {}
                counts["requests"] += 1
                counts["agree"] += action.get("rule_id") in fired if action.get("action") == "refuse" else not fired
    return counts


def first_rule_recorded():
    """The transcoder's Disabled run records the first rule that would act, and the replay must contain it."""
    rule_set, counts = rules("lmsm_transcoder.yaml"), {"requests": 0, "agree": 0}
    for s in SEEDS:
        for r in outputs("combined", "harmbench", s, "disabled"):
            recorded = {k.split("/", 1)[1] for k in r.get("policy_actions") or {} if k.startswith("transcoder/")}
            counts["requests"] += 1
            counts["agree"] += recorded <= fires(rule_set, r["feature_maxima"], "transcoder/")
    return counts


def deployed_rates(backend):
    experiment, arm = BACKENDS[backend]["deployed"]
    return {"asr": summary([100 * statistics.mean(yes(labels.unsafe(r)) for r in scored(experiment, "harmbench", s)[arm].values())
                            for s in SEEDS]),
            "frr": summary([100 * statistics.mean(yes(labels.refusal(r)) for r in scored(experiment, "xstest", s)[arm].values())
                            for s in SEEDS])}


# ---- copyright requests -------------------------------------------------------------------------------------------
def copyright_outputs(backend, part, seed, arm):
    path = RUNS / "copyright" / backend / part / f"seed_{seed}" / arm / "outputs.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@functools.lru_cache(maxsize=None)
def copyright_scored(backend, seed):
    return labels.read_scored(RUNS / "copyright" / backend / "xstest" / f"seed_{seed}" / "judge" / "scored.jsonl")


def copyright_results():
    held = {f"copyright_{i:03d}": kind for i, (_, split, kind) in enumerate(SPLITS["copyright_requests"]) if split == "held_out"}
    out = {"held_out_requests": {kind: sum(k == kind for k in held.values()) for kind in ("book", "lyrics")}}
    for backend in BACKENDS:
        rule_set = {k: v for k, v in rules(BACKENDS[backend]["profile"]).items() if k.endswith("copyright")}
        part = out[backend] = {}
        disabled = [{str(r["request_id"]): fires(rule_set, r["feature_maxima"])
                     for r in copyright_outputs(backend, "requests", s, "disabled")} for s in SEEDS]
        policy = [{str(r["request_id"]): r["execution_record"]["release_outcome"] == "refusal"
                   for r in copyright_outputs(backend, "requests", s, "policy")} for s in SEEDS]
        for kind, other in (("book", "lyrics"), ("lyrics", "book")):
            ids = [rid for rid, k in held.items() if k == kind]

            def share(test):
                return summary([100 * sum(test(i, rid) for rid in ids) / len(ids) for i in range(len(SEEDS))])

            part[kind] = {"own_rule_fires": share(lambda i, rid: f"{kind}_copyright" in disabled[i][rid]),
                          "other_rule_fires": share(lambda i, rid: f"{other}_copyright" in disabled[i][rid]),
                          "policy_refuses": share(lambda i, rid: policy[i][rid])}
        part["reproduced_by_disabled_answers"] = summary(
            [100 * statistics.mean(copyright_classified(backend, s)[rid] for rid in held) for s in SEEDS])

        # XSTest FRR of the deployed copyright policy, and of its replay on Disabled answers under the same settings:
        # the copyright run's own Disabled arm for the transcoder, the SAE's Disabled run for the SAE. A response that
        # WildGuard did not label counts as no refusal (one response in the paper's runs).
        deployed, replayed = [], []
        for s in SEEDS:
            judged = copyright_scored(backend, s)
            deployed.append(100 * statistics.mean(
                r["execution_record"]["release_outcome"] == "refusal" or labels.refusal(judged["policy"][str(r["request_id"])]) == "yes"
                for r in copyright_outputs(backend, "xstest", s, "policy")))
            if backend == "transcoder":
                rows, judged_disabled = copyright_outputs(backend, "xstest", s, "disabled"), judged["disabled"]
            else:
                rows, judged_disabled = outputs("sae_defaults", "xstest", s, "disabled"), scored("sae_defaults", "xstest", s)["disabled"]
            replayed.append(100 * statistics.mean(
                labels.refusal(judged_disabled[str(r["request_id"])]) == "yes" or bool(fires(rule_set, r["feature_maxima"]))
                for r in rows))
        part["xstest_frr_policy"], part["xstest_frr_replayed"] = summary(deployed), summary(replayed)
    return out


def text(results):
    """The quantities that the sentences of Section 6.4 and the appendix state."""
    heat, alone, held = results["figure6"], results["rules_alone"], results["rules_alone_held_out"]
    harmbench = COLUMNS[:-1]

    def own_target_first(backend, rule):
        cells = heat[backend][rule]
        return cells[TARGET[rule]] == max(cells[c] for c in harmbench)

    def cut(configs, backend, rule):
        t = f"asr_{TARGET[rule]}"
        return configs[backend][rule][t]["mean"] / configs[backend]["none"][t]["mean"]

    def added_frr(configs, backend, rule):
        return configs[backend][rule]["frr"]["mean"] - configs[backend]["none"]["frr"]["mean"]

    copyright_ = results["copyright"]
    return {
        "category_rules_firing_most_on_their_target": sum(own_target_first(b, r) for b in BACKENDS for r in TARGET),
        "transcoder_category_rules_largest_asr_ratio": max(cut(alone, "transcoder", r) for r in TARGET),
        "transcoder_category_rules_largest_frr": max(alone["transcoder"][r]["frr"]["mean"] for r in TARGET),
        "transcoder_cybercrime_rule": {"asr_before": alone["transcoder"]["none"]["asr_cybercrime_intrusion"]["mean"],
                                       "asr_after": alone["transcoder"]["cybercrime_intrusion"]["asr_cybercrime_intrusion"]["mean"],
                                       "frr": alone["transcoder"]["cybercrime_intrusion"]["frr"]["mean"]},
        "sae_chemical_rule": {"asr_before": alone["sae"]["none"]["asr_chemical_biological"]["mean"],
                              "asr_after": alone["sae"]["chemical_biological"]["asr_chemical_biological"]["mean"],
                              "added_frr": added_frr(alone, "sae", "chemical_biological")},
        "sae_xstest_firing": {r: heat["sae"][r]["xstest"] for r in ("cybercrime_intrusion", "misinformation_disinformation")},
        "largest_gap_replay_to_deployed": max(abs(results["all_rules"][b]["replayed"][m] - results["all_rules"][b]["deployed"][m]["mean"])
                                              for b in BACKENDS for m in ("asr", "frr")),
        "copyright_rules_added_frr": {f"{b}/{r}": added_frr(alone, b, r) for b in BACKENDS for r in ("book_copyright", "lyrics_copyright")},
        "copyright_own_rule_fires": [min(copyright_[b][k]["own_rule_fires"]["mean"] for b in BACKENDS for k in ("book", "lyrics")),
                                     max(copyright_[b][k]["own_rule_fires"]["mean"] for b in BACKENDS for k in ("book", "lyrics"))],
        "copyright_other_rule_fires_at_most": max(copyright_[b][k]["other_rule_fires"]["mean"] for b in BACKENDS for k in ("book", "lyrics")),
        "copyright_policy_refuses": [min(copyright_[b][k]["policy_refuses"]["mean"] for b in BACKENDS for k in ("book", "lyrics")),
                                     max(copyright_[b][k]["policy_refuses"]["mean"] for b in BACKENDS for k in ("book", "lyrics"))],
        "copyright_policy_frr_minus_replayed": {b: copyright_[b]["xstest_frr_policy"]["mean"] - copyright_[b]["xstest_frr_replayed"]["mean"]
                                                for b in BACKENDS},
        "held_out_transcoder_category_rules_largest_asr_ratio": max(cut(held, "transcoder", r) for r in TARGET),
        "held_out_transcoder_category_rules_largest_frr": max(held["transcoder"][r]["frr"]["mean"] for r in TARGET),
        "held_out_sae_chemical_rule": {"asr_before": held["sae"]["none"]["asr_chemical_biological"]["mean"],
                                       "asr_after": held["sae"]["chemical_biological"]["asr_chemical_biological"]["mean"],
                                       "added_frr": added_frr(held, "sae", "chemical_biological")},
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    results = {"figure6": {}, "rules_alone": {}, "rules_alone_held_out": {}, "held_out_firing_change": {},
               "all_rules": {}, "agreement_with_deployed_policies": {}}
    configs = {"none": set(), **{r: {r} for r in RULES}, "all": set(RULES)}
    for backend in BACKENDS:
        seeds = disabled_requests(backend)
        heat, heat_held = firing(seeds), firing(seeds, held_out=True)
        results["figure6"][backend] = {rule: {col: c["mean"] for col, c in cols.items()} for rule, cols in heat.items()}
        results["held_out_firing_change"][backend] = {
            "own_target": max(abs(heat[r][t]["mean"] - heat_held[r][t]["mean"]) for r, t in TARGET.items()),
            "xstest": max(abs(heat[r]["xstest"]["mean"] - heat_held[r]["xstest"]["mean"]) for r in TARGET)}
        results["rules_alone"][backend] = {name: replay(seeds, active) for name, active in configs.items()}
        results["rules_alone_held_out"][backend] = {name: replay(seeds, active, held_out=True) for name, active in configs.items()}
        replayed = results["rules_alone"][backend]["all"]
        results["all_rules"][backend] = {"replayed": {"asr": replayed["asr"]["mean"], "frr": replayed["frr"]["mean"]},
                                         "deployed": deployed_rates(backend)}
        results["agreement_with_deployed_policies"][backend] = agreement(backend)
    results["agreement_with_deployed_policies"]["transcoder_disabled_first_rule"] = first_rule_recorded()
    calibration = set(SPLITS["artifact_calibration_behaviors"])
    xstest_calibration = set(SPLITS["artifact_calibration_xstest_ids"])
    results["calibration_overlap"] = {
        "harmbench_prompts": sum(r["objective"] in calibration for r in outputs("combined", "harmbench", 0, "disabled")),
        "xstest_prompts": sum(int(r["source_id"]) in xstest_calibration for r in outputs("combined", "xstest", 0, "disabled"))}
    results["copyright"] = copyright_results()
    results["text"] = text(results)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
