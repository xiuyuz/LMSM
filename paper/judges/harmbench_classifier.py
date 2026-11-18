#!/usr/bin/env python3
"""The official HarmBench classifier, the second judge of Section 6.3.

The classifier answers "yes" when a response exhibits the requested behavior. It reads the behavior, and for
contextual behaviors its context, from HarmBench's behavior file, through HarmBench's own prompt templates. Each
released response, reasoning included, is cut to its longest prefix that fits the classifier's 2,048-token context.
Requests whose behavior is not in HarmBench's file (3 behaviors, 6 prompts) are skipped.

Usage: harmbench_classifier.py --harmbench DIR --arm NAME=OUTPUTS.jsonl [--arm ...] --output-dir DIR
DIR is a clone of github.com/centerforaisafety/HarmBench at commit 8e1604d.
"""

import argparse
import ast
import csv
import json
from pathlib import Path

MODEL = "cais/HarmBench-Llama-2-13b-cls"
CONTEXT = 2048
MARGIN = 8  # tokens kept free for re-tokenization


def templates(harmbench):
    tree = ast.parse((harmbench / "eval_utils.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LLAMA2_CLS_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("LLAMA2_CLS_PROMPT not found in eval_utils.py")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harmbench", type=Path, required=True)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=OUTPUTS.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = p.parse_args()

    prompt = templates(a.harmbench)
    with open(a.harmbench / "data/behavior_datasets/harmbench_behaviors_text_all.csv", newline="", encoding="utf-8") as f:
        behaviors = {r["BehaviorID"]: r for r in csv.DictReader(f)}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.truncation_side = "right"

    def render(behavior, generation):
        if "context" in behavior["Tags"].split(", "):
            return prompt["prompt_contextual"].format(behavior=behavior["Behavior"], generation=generation,
                                                      context=behavior["ContextString"])
        return prompt["prompt"].format(behavior=behavior["Behavior"], generation=generation)

    def clip(text, limit):
        return tokenizer.decode(tokenizer.encode(text, max_length=limit, truncation=True), skip_special_tokens=True)

    def n_tokens(text):
        return len(tokenizer.encode(text))

    rows, inputs = [], []
    for spec in a.arm:
        name, _, path = spec.partition("=")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            behavior = behaviors.get(r["objective"])
            if behavior is None:
                continue
            budget = max(CONTEXT - 1 - n_tokens(render(behavior, "")) - MARGIN, 1)
            generation = clip(r["released_output"], budget)
            while n_tokens(render(behavior, generation)) > CONTEXT - 1 and budget > 64:
                budget -= 64
                generation = clip(r["released_output"], budget)
            rows.append({"arm": name, "request_id": str(r["request_id"]), "behavior_id": r["objective"]})
            inputs.append(render(behavior, generation))

    llm = LLM(model=MODEL, tokenizer=MODEL, gpu_memory_utilization=a.gpu_memory_utilization, max_model_len=CONTEXT)
    results = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=1))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with open(a.output_dir / "scored.jsonl", "w", encoding="utf-8") as f:
        for row, result in zip(rows, results, strict=True):
            row["raw_judge_output"] = result.outputs[0].text
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
