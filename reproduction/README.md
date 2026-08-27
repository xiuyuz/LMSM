# LMSM result reproduction

This directory reproduces both the current dense-probe results and the supplied
artifact-backed SAE and transcoder results. All workflows use fixed profiles
and parameter files: they do not annotate data, fit backends, select thresholds,
or modify policies.

## Reproduction coverage

| Result family | Supplied policies | Reproduced outputs |
|---|---|---|
| Central safety and utility | Matched Disabled, LMSM-Checkpoint, LMSM-Anytime | HarmBench and WildJailbreak ASR, XSTest-safe FRR, raw generations, actions, and judge records |
| Artifact-backed backends | LMSM-SAE on Gemma-3-4B-IT, LMSM-Transcoder on Qwen3-4B | Six-category HarmBench generation, category-matched rules, six-rule bundles, raw generated-text judgments, and the final six-row table |
| Temporal behavior | LMSM-Checkpoint, LMSM-Anytime | Complete trigger CDF and trigger/admitted-prefix/avoided-token distributions |
| Runtime behavior | Dense probe and selected transcoder coordinate | Continuous-batch isolation, cross-backend measurements, systems overhead, and rule scaling |

The SAE and transcoder instructions are in
[Reproduce LMSM-SAE and LMSM-Transcoder results](#reproduce-lmsm-sae-and-lmsm-transcoder-results).
They cover the complete tested path: benchmark preparation, all generation
shards, all six judge runs, and final table construction.

The central ASR/FRR workflow has three fixed arms:

- `matched_disabled` (`Matched Disabled`): the Checkpoint monitoring path with actions disabled;
- `lmsm_checkpoint`: `LMSM-Checkpoint` with all provided rules active;
- `lmsm_anytime`: `LMSM-Anytime` with all provided rules active.

## Setup

Install LMSM and the evaluation dependencies from the repository root:

```sh
python -m pip install -e .
python -m pip install -r reproduction/requirements.txt
```

The SAE backend reproduction additionally needs the Hugging Face and SAE
extras:

```sh
python -m pip install -e '.[hf,sae]'
```

Set Qwen3-4B or a local model directory:

```sh
export LMSM_MODEL=Qwen/Qwen3-4B
export LMSM_DATA=/path/to/lmsm-evaluation/data
export LMSM_RESULTS=/path/to/lmsm-evaluation/runs
export LMSM_PYTHON=python
export HF_HOME=/path/to/huggingface/cache
```

All GPU commands require an explicit physical GPU number. The scripts never
select, stop, or replace GPU jobs.

Keep generated data and runs outside the repository checkout so the source tree
stays easy to navigate.

## Prepare evaluation inputs

```sh
python reproduction/prepare_benchmarks.py \
  --output-dir "$LMSM_DATA"
```

This creates:

- `harmbench_602.jsonl` from the ThinkSafe evaluation source;
- `wildjailbreak_harmful_2000.jsonl` from the WildJailbreak evaluation split;
- `xstest_safe_250.jsonl` from the XSTest CSV.
- `xstest_systems_150.jsonl`, the fixed held-out XSTest subset used by the
  systems measurements.

WildJailbreak access must already be available to the local Hugging Face
environment.

## Run generation

Run every benchmark-arm pair. The last argument is the physical GPU number.
For example:

```sh
sh reproduction/run_generation.sh harmbench matched_disabled 0
sh reproduction/run_generation.sh harmbench lmsm_checkpoint 0
sh reproduction/run_generation.sh harmbench lmsm_anytime 0

sh reproduction/run_generation.sh wildjailbreak matched_disabled 0
sh reproduction/run_generation.sh wildjailbreak lmsm_checkpoint 0
sh reproduction/run_generation.sh wildjailbreak lmsm_anytime 0

sh reproduction/run_generation.sh xstest matched_disabled 0
sh reproduction/run_generation.sh xstest lmsm_checkpoint 0
sh reproduction/run_generation.sh xstest lmsm_anytime 0
```

HarmBench and WildJailbreak use nucleus sampling with a 16,384-token maximum.
XSTest uses greedy decoding with a **512-token maximum** and deterministic
per-row seeds beginning at 1,000,000.

Each run directory contains `outputs.jsonl`, `actions.jsonl`, `metadata.json`,
and `timings.csv`. Safety and utility rates use
`released_output`; trigger timing uses the admitted-prefix action record.

## Judge harmful responses

Use Llama-Guard for HarmBench and WildJailbreak:

```sh
python reproduction/judge_harmful.py \
  --benchmark harmbench \
  --arm matched_disabled="$LMSM_RESULTS/harmbench/matched_disabled/outputs.jsonl" \
  --arm lmsm_checkpoint="$LMSM_RESULTS/harmbench/lmsm_checkpoint/outputs.jsonl" \
  --arm lmsm_anytime="$LMSM_RESULTS/harmbench/lmsm_anytime/outputs.jsonl" \
  --output-dir "$LMSM_RESULTS/harmbench/judge" \
  --physical-gpu 0

python reproduction/judge_harmful.py \
  --benchmark wildjailbreak \
  --arm matched_disabled="$LMSM_RESULTS/wildjailbreak/matched_disabled/outputs.jsonl" \
  --arm lmsm_checkpoint="$LMSM_RESULTS/wildjailbreak/lmsm_checkpoint/outputs.jsonl" \
  --arm lmsm_anytime="$LMSM_RESULTS/wildjailbreak/lmsm_anytime/outputs.jsonl" \
  --output-dir "$LMSM_RESULTS/wildjailbreak/judge" \
  --physical-gpu 0
```

HarmBench judges one deterministic blind mix of Disabled, Checkpoint, and
Anytime responses. This three-arm layout differs from the larger batching
context used to produce the retained values, so near-boundary judge counts are
protocol-matched rather than bit-for-bit targets.
WildJailbreak first judges the mixed Checkpoint and Anytime batch on the fresh
engine, then judges Disabled separately, matching the retained guarded-engine
state. The harmful-response judge also keeps vLLM's default engine process
topology used by the retained evaluation. Raw
first-token log probabilities are retained because bfloat16 scores exactly at
the safe/unsafe boundary can change by a few labels across GPU executions even
when the ordered judge inputs are identical.

## Judge XSTest refusals

```sh
python reproduction/judge_refusal.py \
  --arm matched_disabled="$LMSM_RESULTS/xstest/matched_disabled/outputs.jsonl" \
  --arm lmsm_checkpoint="$LMSM_RESULTS/xstest/lmsm_checkpoint/outputs.jsonl" \
  --arm lmsm_anytime="$LMSM_RESULTS/xstest/lmsm_anytime/outputs.jsonl" \
  --output-dir "$LMSM_RESULTS/xstest/judge" \
  --physical-gpu 0
```

Both judge scripts retain the raw judge output and the parsed binary label for
every response.

## Build the tables

```sh
python reproduction/build_tables.py \
  --harmbench-summary "$LMSM_RESULTS/harmbench/judge/summary.json" \
  --wildjailbreak-summary "$LMSM_RESULTS/wildjailbreak/judge/summary.json" \
  --xstest-summary "$LMSM_RESULTS/xstest/judge/summary.json" \
  --output-dir "$LMSM_RESULTS/tables"
```

The command writes `main_safety_utility.csv`, `binary_rate_wilson95.csv`, and
`xstest_decomposition.csv` using only the completed judge summaries.

## Reproduce trigger timing

The temporal figure uses a separate matched greedy run with a 512-token cap
and deterministic per-row seeds beginning at 4,000,000:

```sh
sh reproduction/run_temporal.sh matched_disabled 0
sh reproduction/run_temporal.sh lmsm_checkpoint 0
sh reproduction/run_temporal.sh lmsm_anytime 0

python reproduction/build_temporal_tables.py \
  --disabled-outputs "$LMSM_RESULTS/temporal/matched_disabled/outputs.jsonl" \
  --checkpoint-actions "$LMSM_RESULTS/temporal/lmsm_checkpoint/actions.jsonl" \
  --anytime-actions "$LMSM_RESULTS/temporal/lmsm_anytime/actions.jsonl" \
  --output-dir "$LMSM_RESULTS/temporal/tables"
```

This writes the exact empirical trigger CDF and the trigger, admitted-prefix,
and avoided-token distributions used by the temporal figure.

## Reproduce continuous-batch isolation

The scheduler-churn experiment runs 32 duplicate prompt pairs in two waves with
per-request generation caps of 8, 16, 24, and 32 tokens:

```sh
python reproduction/run_batch_isolation.py \
  --model "$LMSM_MODEL" \
  --profile profiles/lmsm_anytime.yaml \
  --xstest-input "$LMSM_DATA/xstest_safe_250.jsonl" \
  --run-dir "$LMSM_RESULTS/batch_isolation" \
  --physical-gpu 0
```

The run writes `summary.json`, `batch_events.json`, and
`duplicate_pair_margins.jsonl`. The last file retains every duplicate pair's
relevant rule, threshold, two scores, signed and absolute threshold margins,
observed score difference, decision agreement, and certification verdict. A
pair is margin-certified when

```text
min(|score_wave0 - threshold|, |score_wave1 - threshold|)
    > |score_wave0 - score_wave1|.
```

For an action pair, the relevant rule is the selected action rule. For a
non-action pair, it is the rule closest to its threshold. The retained run has
32/32 exact decisions, 32/32 exact threshold-crossing vectors, and 31/32
margin-certified pairs. The sole uncertified pair is
`batch_isolation_harmful_05`.

Recompute these counts directly from the bundled per-pair values without a GPU:

```sh
python reproduction/check_batch_isolation_margins.py
```

## Reproduce the cross-backend monitor

The cross-backend table uses the first wave of the batch-isolation workload:
16 harmful and 16 benign prompts. Both arms use the same Qwen3-4B layer-24
MLP-input hook, request-keyed runtime, monitor-only policy, greedy decoding,
and 64 model tokens per prompt. Only the backend binding changes.

```sh
sh reproduction/run_cross_backend.sh \
  dense_probe 0 "$LMSM_RESULTS/batch_isolation/inputs.jsonl"
sh reproduction/run_cross_backend.sh \
  transcoder 0 "$LMSM_RESULTS/batch_isolation/inputs.jsonl"

python reproduction/build_cross_backend_table.py \
  --run-root "$LMSM_RESULTS/cross_backend" \
  --output "$LMSM_RESULTS/cross_backend/cross_backend.csv"
```

The dense arm uses the supplied standardized chemical/biological dense probe.
The transcoder arm loads coordinate 43,497 from the supplied Qwen layer-24
transcoder. Each arm performs one warmup and three measured repetitions with
actions disabled. The expected table is
`reproduction/reference/expected_cross_backend.csv`; throughput is a wall-clock
measurement, while prompt count, fixed token count, finite scores, backend
evaluation count, and zero actions should match exactly.

## Reproduce systems measurements

Run the configurations needed for matched overhead and rule-count scaling:

```sh
for configuration in \
  matched_native_start_seq1 matched_empty_extension_seq1 \
  lmsm_checkpoint_seq1 lmsm_anytime_seq1 \
  matched_native_end_seq1 \
  matched_native_start_seq32 matched_empty_extension_seq32 \
  lmsm_checkpoint_seq32 lmsm_anytime_seq32 \
  matched_native_end_seq32 \
  lmsm_checkpoint_1rule_seq32 lmsm_checkpoint_6rule_seq32 \
  lmsm_checkpoint_15rule_seq32 \
  lmsm_anytime_1rule_seq32 lmsm_anytime_6rule_seq32 \
  compiled_native_seq1 compiled_native_seq32
do
  sh reproduction/run_systems.sh "$configuration" 0
done

python reproduction/build_systems_tables.py \
  --run-root "$LMSM_RESULTS/systems" \
  --output-dir "$LMSM_RESULTS/systems/tables"
```

Each configuration performs one warmup and five measured repetitions. The
summarizer writes `systems_overhead.csv`, `rule_scaling.csv`, and the combined
raw `checkpoint_rule_scaling_repetitions.csv`. It also writes
`compiled_native_ceiling.csv`, which compares compiled native vLLM with the
bracketed eager-native measurements. The individual `timings.csv` files retain
every repetition.

## Reproduce LMSM-SAE and LMSM-Transcoder results

The SAE/transcoder evaluation is a separate historical experiment on a
six-category, 264-row HarmBench split. It is not the 602-row HarmBench protocol
used by the central Checkpoint and Anytime table.

This path was validated end to end for both backends: 792 generated responses
and 792 raw/parsed judge records for LMSM-SAE, plus the same for
LMSM-Transcoder. `reference/reported_backend_results.csv` records the retained
six-row result for comparison with a fresh sampled run.

Prepare the exact split and its six category-routed inputs:

```sh
python reproduction/prepare_backend_benchmark.py \
  --output-dir "$LMSM_DATA/backend_results"

export LMSM_BACKEND_DATA="$LMSM_DATA/backend_results"
export LMSM_BACKEND_RESULTS="$LMSM_RESULTS/backend_results"
```

The preparation loads the contextual, copyright, and standard configurations
of `walledai/HarmBench`, shuffles each with seed 112, performs the retained
category-stratified 25/75 split, and selects the six categories covered by the
provided policies. Benchmark category names remain in `source_category`;
`matched_rule` records the corresponding public rule name.

Generation uses the provided policies without fitting or calibration. It uses
Hugging Face Transformers, bfloat16, temperature 0.6, top-p 0.95, and at most
8,192 new tokens. Guarded runs have no top-k cutoff; unguarded runs inherit the
model's generation configuration. The full bundle uses 12 modulo shards; the
unguarded and category-matched configurations use two. Qwen guarded runs enable
thinking explicitly; Qwen unguarded runs use the tokenizer's default chat
template, matching the retained protocol.

The public runner uses a simple index-based seed sequence. The retained
artifact-backed evaluation used a different seed assignment, so the
nucleus-sampled guarded counts are protocol-matched rather than expected to be
bit-for-bit identical. Every run retains its responses, actions, seeds, and
judge outputs for direct comparison.

```sh
for backend in sae transcoder; do
  for shard in 0 1; do
    sh reproduction/run_backend_generation.sh "$backend" unguarded "$shard" 0
  done

  for rule in \
    book_copyright lyrics_copyright chemical_biological \
    cybercrime_intrusion illegal_activity misinformation_disinformation
  do
    for shard in 0 1; do
      sh reproduction/run_backend_generation.sh \
        "$backend" one_rule "$shard" 0 "$rule"
    done
  done

  for shard in 0 1 2 3 4 5 6 7 8 9 10 11; do
    sh reproduction/run_backend_generation.sh "$backend" six_rules "$shard" 0
  done
done
```

Every shard saves released outputs, interventions, readable run metadata, and
per-request timings. Independent shards may be assigned to different free GPUs
by changing the final GPU argument.

These rows used generated-text classification with
`meta-llama/Llama-Guard-3-8B`: greedy judge generation, at most 16 tokens, then
parsing `safe` or `unsafe`. This differs from the first-token log-probability
judge used by the central table. Judge the six configurations as follows:

```sh
python reproduction/judge_backend_results.py \
  --run-root "$LMSM_BACKEND_RESULTS/sae/unguarded" \
  --model-label Gemma-3-4B-IT --method LMSM-SAE \
  --configuration Unguarded \
  --output-dir "$LMSM_BACKEND_RESULTS/judge/sae_unguarded" --physical-gpu 0 \
  --batch-size 16
python reproduction/judge_backend_results.py \
  --run-root "$LMSM_BACKEND_RESULTS/sae/one_rule" \
  --model-label Gemma-3-4B-IT --method LMSM-SAE \
  --configuration "Category-Matched Rule" \
  --output-dir "$LMSM_BACKEND_RESULTS/judge/sae_category_matched" --physical-gpu 0
python reproduction/judge_backend_results.py \
  --run-root "$LMSM_BACKEND_RESULTS/sae/six_rules" \
  --model-label Gemma-3-4B-IT --method LMSM-SAE \
  --configuration "Six-Rule Bundle" \
  --output-dir "$LMSM_BACKEND_RESULTS/judge/sae_six_rule_bundle" --physical-gpu 0

python reproduction/judge_backend_results.py \
  --run-root "$LMSM_BACKEND_RESULTS/transcoder/unguarded" \
  --model-label Qwen3-4B --method LMSM-Transcoder \
  --configuration Unguarded \
  --output-dir "$LMSM_BACKEND_RESULTS/judge/transcoder_unguarded" --physical-gpu 0 \
  --batch-size 8
python reproduction/judge_backend_results.py \
  --run-root "$LMSM_BACKEND_RESULTS/transcoder/one_rule" \
  --model-label Qwen3-4B --method LMSM-Transcoder \
  --configuration "Category-Matched Rule" \
  --output-dir "$LMSM_BACKEND_RESULTS/judge/transcoder_category_matched" --physical-gpu 0
python reproduction/judge_backend_results.py \
  --run-root "$LMSM_BACKEND_RESULTS/transcoder/six_rules" \
  --model-label Qwen3-4B --method LMSM-Transcoder \
  --configuration "Six-Rule Bundle" \
  --output-dir "$LMSM_BACKEND_RESULTS/judge/transcoder_six_rule_bundle" --physical-gpu 0
```

Each judge directory retains the raw generated judge text and parsed label for
all 264 responses. Build the final table from the six summaries:

```sh
python reproduction/build_backend_results_table.py \
  --summary "$LMSM_BACKEND_RESULTS/judge/sae_unguarded/summary.json" \
  --summary "$LMSM_BACKEND_RESULTS/judge/sae_category_matched/summary.json" \
  --summary "$LMSM_BACKEND_RESULTS/judge/sae_six_rule_bundle/summary.json" \
  --summary "$LMSM_BACKEND_RESULTS/judge/transcoder_unguarded/summary.json" \
  --summary "$LMSM_BACKEND_RESULTS/judge/transcoder_category_matched/summary.json" \
  --summary "$LMSM_BACKEND_RESULTS/judge/transcoder_six_rule_bundle/summary.json" \
  --output "$LMSM_BACKEND_RESULTS/backend_results.csv"
```

The category-matched row concatenates six evaluations in which only the prompt
category's corresponding rule is active. It is not one universal single-rule
policy. The split book/lyrics copyright rules preserve the trigger behavior of
the two single-feature copyright backends used in the retained experiment.

## Expected evaluation outputs

The small CSV files in `reference/` record expected outputs for the fixed
configuration. They provide convenient targets when checking a reproduction;
they are not inputs to generation or judging.

- `reported_rates.csv` contains the nine reported central binary rates.
- `expected_batch_isolation.csv` contains the scheduler-churn invariants.
- `batch_isolation_pair_margins.jsonl` contains all 32 retained per-pair
  threshold distances, observed differences, and certification verdicts.
- `expected_cross_backend.csv` contains the monitor-only dense-probe and
  selected-coordinate backend results.
- `expected_systems_overhead.csv` contains the available matched throughput
  measurements.
- `expected_rule_scaling.csv` contains the available rule-count measurements.
- `expected_compiled_native.csv` contains the optimized native ceiling.
- `checkpoint_rule_scaling_repetitions.csv` contains all five raw Checkpoint
  throughput repetitions for 0, 1, 6, and 15 active rules.
- `reported_backend_results.csv` contains the six retained SAE/transcoder
  HarmBench counts and rates for comparison with a fresh sampled run.
- `trigger_cdf.csv` and `trigger_distributions.csv` contain the complete
  temporal figure data.

Generated tokens, action counts, XSTest refusal counts, and trigger tables
should match exactly. The bfloat16 first-token harmful-response judge can vary
on near-tied rows, and the sampled artifact-backed SAE/transcoder counts should
reproduce the reported effect rather than identical counts. Throughput is a
wall-clock measurement and should preserve the reported ordering and overhead
scale rather than reproduce identical floating-point timings.
