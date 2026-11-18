# Reproducing the evaluation

This folder reproduces the tables, figures and numbers of the paper's evaluation (Section 6) and its appendix.
Each subfolder covers one subsection.
It holds the scripts that ran the experiments, the analysis that turns their outputs into the reported numbers,
and those numbers as `results.json`.

| Subsection | Folder | Tables and figures |
|---|---|---|
| 6.2 Mediation correctness | [`mediation/`](mediation/README.md) | Table 4, Figure 5 |
| 6.3 Policy effectiveness | [`effectiveness/`](effectiveness/README.md) | Tables 5 and 6 |
| 6.4 Released artifacts and rule control | [`released_artifacts/`](released_artifacts/README.md) | Figure 6 |
| 6.5 Serving cost | [`cost/`](cost/README.md) | Figure 7 |

[`judges/`](judges/README.md) labels the responses, and [`data/`](data/README.md) rebuilds the benchmark inputs.
`common.sh` holds the paths and settings that every run script shares.
Each folder's README maps every number of its subsection to a key in its `results.json`.

## Three ways to check a result

1. **Print the numbers** behind every table and figure from the shipped results. This needs only Python.
   ```bash
   python paper/show.py                        # every number
   python paper/show.py effectiveness table5   # one table
   ```
2. **Recompute the numbers** from run outputs. Each folder's `analyze.py` or `summarize.py` reads the runs below
   `paper/runs/` and writes its `results.json`.
3. **Rerun the experiments.** Each folder's README lists the commands and the time each took.
   All runs together take about 16 hours on one NVIDIA H100 GPU, plus model loading and judging.

## Setup for rerunning

```bash
python -m pip install -e '.[hf,vllm,sae]'
```

The runs download these models from the Hugging Face Hub.

- **Served models:** Qwen/Qwen3-4B and google/gemma-3-4b-it. The serving-cost runs also use Qwen/Qwen3-8B,
  Qwen/Qwen3-14B and Qwen/Qwen3-32B.
- **Released artifacts:** the transcoder mwhanna/qwen3-4b-transcoders and the Gemma Scope SAE
  gemma-scope-2-4b-it-res. The profiles name both.
- **Judges and the output filter:** meta-llama/Llama-Guard-3-8B, allenai/wildguard and
  cais/HarmBench-Llama-2-13b-cls.

`data/README.md` explains how to rebuild the benchmark inputs.
The copyright judge also needs a clone of HarmBench, `python -m pip install spacy datasketch` and
`python -m spacy download en_core_web_sm`.
Set `QWEN`, `GEMMA`, `DATA`, `RUNS` or `GPU` to change the defaults in `common.sh`.

## Not included

Model responses are not included, because many of them are harmful completions.
The run scripts regenerate them.
The benchmark prompts are not included either, and `data/` rebuilds them from their public sources.
