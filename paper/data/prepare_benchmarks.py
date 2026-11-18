#!/usr/bin/env python3
"""Rebuild the benchmark inputs of the paper from their public sources.

Writes into OUT_DIR, one JSON line per request in the order the runs used:
  harmbench_602.jsonl               ThinkSafe's 602 HarmBench prompts
  wildjailbreak_harmful_2000.jsonl  the 2,000 adversarial harmful prompts of WildJailbreak's evaluation set
  xstest_safe_250.jsonl             the 250 safe XSTest prompts
  xstest_systems_150.jsonl          150 of them, for the serving-cost runs
  strongreject_313.jsonl            the 313 StrongREJECT prompts
  ordinary_3042.jsonl               3,042 ordinary requests from No Robots, GSM8K, HumanEval+ and MBPP+
  copyright_100.jsonl               the 100 HarmBench copyright behaviors (needs --harmbench)
WildJailbreak and No Robots need a Hugging Face login with their terms accepted.

Usage: prepare_benchmarks.py --out-dir DIR [--harmbench HARMBENCH_CLONE]
"""

import argparse
import ast
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
THINKSAFE = "https://raw.githubusercontent.com/seanie12/ThinkSafe/3ff6367873b5781de9abfe1bca5f17dc4950e5c0/dataset/eval/harmbench.json"
XSTEST = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
STRONGREJECT = ("Machlovi/strongreject-dataset", "c18bb810edc4b60b815b205faeaa2bd5f72c3e93")
ORDINARY_SEED = 20261113
CODE_INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem in a "
                    "markdown code block:")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")


def load(name, *args, **kwargs):
    import datasets
    return datasets.load_dataset(name, *args, **kwargs)


def harmbench():
    return [{**row, "id": f"harmbench_{i:03d}", "source": "seanie12/ThinkSafe", "split": "dataset/eval/harmbench.json",
             "source_row": i, "label": "harmful", "safety_label": "harmful"}
            for i, row in enumerate(json.loads(fetch(THINKSAFE)))]


def wildjailbreak():
    rows = []
    data = load("allenai/wildjailbreak", "eval", split="train", delimiter="\t", keep_default_na=False)
    for i, item in enumerate(map(dict, data)):
        if item.get("data_type") == "adversarial_harmful":
            rows.append({**item, "id": f"wildjailbreak_harmful_{len(rows):04d}", "source": "allenai/wildjailbreak",
                         "split": "eval/train", "source_row": i, "source_label": item.get("label"),
                         "prompt": item["adversarial"], "label": "harmful", "safety_label": "harmful"})
    return rows


def xstest():
    rows = []
    for i, row in enumerate(csv.DictReader(io.StringIO(fetch(XSTEST)))):
        if row["label"].strip().lower() == "safe":
            rows.append({**row, "id": f"xstest_safe_{len(rows):04d}", "source": "paul-rottger/xstest", "split": "test",
                         "source_row": i, "source_id": row["id"], "xstest_label": "safe", "label": "benign",
                         "safety_label": "benign"})
    return rows


def strongreject():
    name, revision = STRONGREJECT
    return [{"id": f"strongreject_{i:03d}", "prompt": r["prompt"], "label": "harmful", "safety_label": "harmful",
             "benchmark": "strongreject", "category": r["category"], "source": r["source"],
             "prompt_label": r["prompt_label"], "dataset": name, "revision": revision, "split": "train", "source_row": i}
            for i, r in enumerate(load(name, split="train", revision=revision))]


def ordinary():
    def row(rid, prompt, dataset, category, source, split, source_row, source_id, construction):
        return {"id": rid, "prompt": prompt, "label": "benign", "safety_label": "benign", "dataset": dataset,
                "category": category, "source": source, "split": split, "source_row": source_row,
                "source_id": source_id, "construction": construction}

    def code(task):
        return f"{CODE_INSTRUCTION}\n```\n{task.strip()}\n```\n"

    def first_turn(messages):
        system = [m["content"] for m in messages if m["role"] == "system"]
        user = [m["content"] for m in messages if m["role"] == "user"][0]
        if system:
            return f"{system[0].strip()}\n\n{user.strip()}", "system persona + first user turn"
        return user, "first user turn"

    out = []
    no_robots = load("HuggingFaceH4/no_robots")
    for i, r in enumerate(no_robots["test"]):
        text, how = first_turn(r["messages"])
        out.append(row(f"norobots_test_{i:05d}", text, "no_robots", r["category"], "HuggingFaceH4/no_robots",
                       "test", i, r["prompt_id"], how))
    by_category = {}
    for i, r in enumerate(no_robots["train"]):
        by_category.setdefault(r["category"], []).append(i)
    for category in sorted(by_category):
        rng = random.Random(f"{ORDINARY_SEED}:no_robots:{category}")
        for i in sorted(rng.sample(by_category[category], min(150, len(by_category[category])))):
            r = no_robots["train"][i]
            text, how = first_turn(r["messages"])
            out.append(row(f"norobots_train_{i:05d}", text, "no_robots", r["category"], "HuggingFaceH4/no_robots",
                           "train", i, r["prompt_id"], how))
    gsm8k = load("openai/gsm8k", "main")["test"]
    for i in sorted(random.Random(f"{ORDINARY_SEED}:gsm8k").sample(range(len(gsm8k)), 500)):
        out.append(row(f"gsm8k_test_{i:04d}", gsm8k[i]["question"], "gsm8k", "math_word_problem", "openai/gsm8k",
                       "test", i, str(i), "question text"))
    for i, r in enumerate(load("evalplus/humanevalplus")["test"]):
        out.append(row(f"humanevalplus_{i:03d}", code(r["prompt"]), "humanevalplus", "code_completion",
                       "evalplus/humanevalplus", "test", i, r["task_id"], "EvalPlus instruction + function stub"))
    for i, r in enumerate(load("evalplus/mbppplus")["test"]):
        tests = ast.literal_eval(r["test_list"]) if isinstance(r["test_list"], str) else list(r["test_list"])
        out.append(row(f"mbppplus_{i:03d}", code(f'"""\n{r["prompt"]}\n{tests[0]}\n"""\n'), "mbppplus",
                       "code_from_description", "evalplus/mbppplus", "test", i, str(r["task_id"]),
                       "EvalPlus instruction + MBPP docstring with first test"))
    return out


def copyright_requests(harmbench_clone):
    path = harmbench_clone / "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    with open(path, newline="", encoding="utf-8") as f:
        behaviors = {r["BehaviorID"]: r for r in csv.DictReader(f)}
    splits = json.loads((HERE / "splits.json").read_text())["copyright_requests"]
    rows = []
    for i, (behavior_id, split, kind) in enumerate(splits):
        b = behaviors[behavior_id]
        rows.append({"id": f"copyright_{i:03d}", "prompt": b["Behavior"], "split": split, "behavior_id": behavior_id,
                     "tags": b["Tags"], "kind": kind})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--harmbench", type=Path, help="a clone of github.com/centerforaisafety/HarmBench")
    a = p.parse_args()

    safe = xstest()
    systems = set(json.loads((HERE / "splits.json").read_text())["xstest_systems_source_ids"])
    files = {"harmbench_602.jsonl": harmbench(),
             "wildjailbreak_harmful_2000.jsonl": wildjailbreak(),
             "xstest_safe_250.jsonl": safe,
             "xstest_systems_150.jsonl": [r for r in safe if int(r["source_id"]) in systems],
             "strongreject_313.jsonl": strongreject(),
             "ordinary_3042.jsonl": ordinary()}
    if a.harmbench:
        files["copyright_100.jsonl"] = copyright_requests(a.harmbench)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        expected = int(name.rsplit("_", 1)[1].split(".")[0])
        if len(rows) != expected:
            raise SystemExit(f"{name}: {len(rows)} rows, expected {expected}")
        with open(a.out_dir / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"wrote {len(rows):5d} rows to {a.out_dir / name}")


if __name__ == "__main__":
    main()
