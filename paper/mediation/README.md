# Mediation correctness (Section 6.2, Table 4, Figure 5)

This folder tests whether LMSM performs its three duties in compiled vLLM, independently of how well a policy recognizes harmful output.

## Files

| File | What it does |
|---|---|
| `make_workload.py` | Builds the association workload, each of the 852 HarmBench and XSTest prompts submitted twice in a seeded order. |
| `association.py` | Runs the workload through `examples/run_vllm.py` and records batch updates, layouts, evidence and per-step state. |
| `association.sh` | The association runs, 14 with the probe policies and 8 with the transcoder and the SAE, each followed by its check. |
| `check_probe_run.py` | Checks one probe run: each decision follows from the request's own evidence, and each admission starts fresh. |
| `check_feature_run.py` | The same checks for the transcoder and the SAE, from the feature engine's recorded state. |
| `ablation.sh` | Keys policy state by batch row, with and without the reset at admission, against the request-keyed control. |
| `contract_tests.sh` | The startup check, a removed capture, a dropped observation, and device against host evaluation. |
| `choose_drop_targets.py` | Picks the 8 requests per policy whose observation the dropped-observation test removes. |
| `not_reached.sh` | Short answers under Checkpoint, released with a record or refused, to test the not-reached outcome. |
| `run.sh` | Runs everything above, then `summarize.py`, and prints the numbers. |
| `summarize.py` | Turns the runs under `paper/runs/mediation` into `results.json`. |
| `results.json` | The numbers of Table 4, Figure 5 and Section 6.2, from the paper's runs. |

## Commands

Run these from the repository root.

```bash
bash paper/mediation/run.sh             # every test, then results.json and its numbers
python paper/mediation/summarize.py     # results.json from the runs under paper/runs
python paper/show.py mediation          # the numbers of Table 4, Figure 5 and Section 6.2
```

`summarize.py` counts the released-artifact requests from the runs of `paper/released_artifacts`, so run those first.

## GPU time

The paper's runs took about these times on one NVIDIA H100 GPU, counted from each run's start to its end.

| Experiment | Runs | Time |
|---|---|---|
| `association.sh`, probe policies | 14 | 0.5 h |
| `association.sh`, transcoder and SAE | 8 | 0.4 h |
| `ablation.sh` | 16 | 1.3 h |
| `contract_tests.sh` | 27 | 12 min |
| `not_reached.sh` | 6 | 4 min of generation, plus model loading |

## Where each number comes from

All keys are in `results.json`.

| Paper | Key |
|---|---|
| Table 4, probe policies, 23,856 | `association.probes.requests` and `association_errors` |
| Table 4, released artifacts, 13,632 | `association.released_artifacts.requests` and `association_errors` |
| "Across 37,488 request executions on the three backends" | `association.requests` |
| Table 4, duplicate pairs, 5,964, and "all 5,964 duplicate pairs" | `duplicate_pairs.batch_invariant` |
| "0.8 to 6.3% of the pairs under the probe policies" | `duplicate_pairs.default_kernels.differing_pct` |
| Figure 5, each tile | `ablation.<policy>/<rowkey_reset or rowkey_noreset>/<arrival>`, the count `used_another_requests_state` for Anytime and `failed_closed` for Checkpoint |
| Figure 5, the request-keyed controls | `ablation.<policy>/control/<arrival>.control_states` |
| "904 of its 1,704 decisions change, including 878 HarmBench requests" | `ablation.anytime/rowkey_noreset/burst`, `decisions_changed` and `harmbench_refusals_released` |
| Table 4, unsupported configurations | `contract.refused_at_startup` |
| Table 4, capture removed, 384 | `contract.capture_removed` and `capture_removed_failed_closed` |
| Table 4, one observation dropped, 32 | `contract.observation_dropped` and `observation_dropped_failed_closed` |
| Table 4, device versus host, 128 | `contract.device_vs_host` and `device_vs_host_same` |
| Table 4, failed requests, 416 | `contract.failed_requests` and `failed_requests_fixed_refusal` |
| Table 4, not reached on short answers, 5,704 | `not_reached.requests` and `mismatches` |
| Table 4, released-artifact policies, 7,668 | `released_as_decided.requests` and `mismatches` |
| "each of the 14 runs ... reuses the rows of finished requests more than 1,500 times" (the appendix) | `association.probes.fewest_slot_reuses` |
| "four of them enable chunked prefill and prefix caching" (the appendix) | `association.probes.runs_with_chunked_prefill_and_prefix_caching` |

The paper's 5,964 duplicate pairs come from seven control runs, because three controls ran twice.
`ablation.sh` runs each of the four controls once, which gives 3,408 pairs.
