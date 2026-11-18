# Policy effectiveness (Section 6.3)

This folder reproduces Tables 5 and 6, the numbers of Section 6.3, and the appendix paragraph on generation and
judging. The probe policies are Checkpoint and Anytime on Qwen3-4B, and Disabled runs the Checkpoint path with its
actions suppressed.

## Numbers from results (CPU, seconds)

`results.json` holds every number, as `analyze.py` computed it from the paper's runs.

```bash
python paper/show.py effectiveness          # every number of Section 6.3
python paper/show.py effectiveness table5   # Table 5 alone
```

## Results from runs (CPU, minutes)

```bash
python paper/effectiveness/analyze.py      # reads $RUNS/effectiveness and writes results.json
```

## Runs (GPU)

Prepare the inputs first (`../data/README.md`). Every run uses one H100 GPU, and `settings.sh` holds the
generation settings: nucleus sampling at temperature 0.6, top-p 0.95, top-k 20, up to 16,384 new tokens, and up to
32 concurrent requests. Matched settings use one synchronous engine process without prefix caching or chunked
prefill. The times are the generation time of the paper's runs, without model loading and judging.

| Command | Runs | Time |
|---|---|---|
| `paper/effectiveness/run.sh matched harmbench`, then `wildjailbreak` and `xstest` | Table 5 under matched settings | 2.4 h |
| `paper/effectiveness/run.sh defaults harmbench`, then `wildjailbreak` and `xstest` | the same with vLLM's default optimizations | 2.1 h |
| `paper/effectiveness/run.sh defaults strongreject` | the StrongREJECT column of Table 5 | 0.2 h |
| `ARMS=checkpoint paper/effectiveness/run.sh defaults ordinary` | the ordinary requests | 1.3 h |
| `ARMS=disabled paper/effectiveness/run.sh defaults ordinary 0` | Disabled answers for the timing workload | 0.4 h |
| `HARMBENCH=/path/to/HarmBench paper/effectiveness/second_judge.sh` | the official HarmBench classifier | judging only |
| `paper/effectiveness/filter.sh` | the Llama Guard 3 filter and the WildGuard labels of Table 6 | judging only |
| `paper/effectiveness/filter_timing.sh` | the time column of Table 6, alone on an idle GPU | 0.5 h |

Each run writes `outputs.jsonl` with one row per request and its execution record. The layout is
`$RUNS/effectiveness/{matched,defaults}/BENCHMARK/seed_S/{disabled,checkpoint,anytime,judge}`, with the filter
runs under `filter/` and the timing session under `filter_timing/`.

## Where each number comes from

| In the paper | Key in `results.json` |
|---|---|
| Table 5, LMSM rows | `table5.{disabled,checkpoint,anytime}.{harmbench,strongreject,wildjailbreak,xstest,mean}` |
| Table 5, ThinkSafe rows | `table5_reported_by_thinksafe`, copied from Table 1 of ThinkSafe |
| Table 6 | `table6.{filter,checkpoint,combined}.{harmbench,wildjailbreak,xstest,time_added}` |
| HarmBench ASR from 39.4% to 3.3%, XSTest FRR from 0.8% to 3.1% | `table5.disabled`, `table5.checkpoint` |
| WildJailbreak ASR reduced by a factor of six | `text.wildjailbreak_reduction_factor` |
| each LMSM rate varies by at most 0.8 points across seeds | `text.largest_seed_range_of_an_lmsm_rate` |
| rates change by at most 0.7 points with vLLM's default optimizations | `text.largest_change_with_vllm_defaults`, from `vllm_defaults` |
| Anytime acts at a median of 14 generated tokens instead of 64 | `intervention_steps_matched` |
| the model without safety training is within 2 points of Disabled | `text.largest_gap_no_safety_training_to_disabled` |
| both LMSM policies reach a lower mean of the four rates | `table5.{checkpoint,anytime}.mean`, `text.lowest_reported_mean` |
| 224 never-seen HarmBench prompts, from 36.2% to 2.2% | `never_seen` |
| the HarmBench classifier rates 50.1% and 8.4% as harmful | `harmbench_classifier` |
| 3,042 ordinary requests, 2.2%, almost all No Robots | `ordinary_requests` |
| harmful releases about half as often as the filter | `text.checkpoint_over_filter_harmful_releases` |
| less than half of the filter's time | `text.checkpoint_over_filter_time` |
| an FRR 2 points higher | `text.checkpoint_minus_filter_frr` |
| both together release 30 to 40% of what Checkpoint releases | `text.combined_over_checkpoint_harmful_releases` |
| both together add 0.3 points of FRR | `text.combined_minus_checkpoint_frr` |
| both together take about as long as the filter alone | `table6.combined.time_added`, `table6.filter.time_added` |
| at most 1.3% of Table 6's responses without a label | `table6_details.largest_unlabeled_share` |
| WildGuard judges the 41 new pairs | `table6_details.new_pairs` |
| 128 responses, 32 from each source, 52 of them the fixed refusal | `table6_details.timing_*` |
