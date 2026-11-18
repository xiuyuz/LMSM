# Benchmark inputs

`prepare_benchmarks.py` rebuilds every input file of the experiments from its public source. We do not ship the
prompts themselves, since WildJailbreak and No Robots ask users to accept their terms first.

```bash
huggingface-cli login            # with the terms of allenai/wildjailbreak and HuggingFaceH4/no_robots accepted
git clone https://github.com/centerforaisafety/HarmBench /path/to/HarmBench
python paper/data/prepare_benchmarks.py --out-dir paper/data/benchmarks --harmbench /path/to/HarmBench
```

The experiment scripts read the files from `paper/data/benchmarks`, or from `$DATA` when it is set. Each file
matches the file our runs read byte for byte, except that the copyright file numbers its request IDs anew.

| File | Requests | Source |
|---|---|---|
| `harmbench_602.jsonl` | 602 | ThinkSafe's HarmBench prompts (github.com/seanie12/ThinkSafe) |
| `wildjailbreak_harmful_2000.jsonl` | 2,000 | WildJailbreak, adversarial harmful prompts of the evaluation set |
| `xstest_safe_250.jsonl` | 250 | XSTest, the safe prompts (github.com/paul-rottger/xstest) |
| `xstest_systems_150.jsonl` | 150 | the XSTest prompts of the serving-cost runs |
| `strongreject_313.jsonl` | 313 | StrongREJECT (Machlovi/strongreject-dataset) |
| `ordinary_3042.jsonl` | 3,042 | No Robots, GSM8K, HumanEval+ and MBPP+ |
| `copyright_100.jsonl` | 100 | the HarmBench copyright behaviors |

The ordinary requests are all 500 No Robots test prompts, 150 No Robots training prompts per category, 500 GSM8K
test problems, and all HumanEval+ and MBPP+ tasks. A fixed seed chooses the samples.

`splits.json` holds fixed lists that the analyses need.

| Key | Contents |
|---|---|
| `xstest_systems_source_ids` | the XSTest IDs of the serving-cost prompts |
| `copyright_requests` | the HarmBench behavior, split and kind of each copyright request, in file order |
| `development_behaviors` | the HarmBench behaviors that appeared during development (Section 6.3, "never appeared") |
| `artifact_calibration_behaviors` | the HarmBench behaviors that the search for the released-artifact rules could draw on |
| `artifact_calibration_xstest_ids` | the 100 XSTest prompts that the same search could draw on |
| `artifact_held_out_xstest_ids` | the other 150 XSTest prompts |
| `harmbench_categories` | the HarmBench category of each behavior |
