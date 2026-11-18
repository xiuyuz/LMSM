# Serving cost (Section 6.5, Figure 7)

This folder measures the throughput of each deployment relative to native vLLM serving the same model in the same session.

## Files

| File | What it does |
|---|---|
| `arm.sh` | Runs one arm at one batch width with `examples/run_vllm.py`. |
| `session.sh` | Runs every arm at each width in two rounds, the second in reverse order. Each arm starts after the GPU has been idle for 20 seconds, and a sampler records any other process on the GPU. |
| `synthetic_profile.py` | Writes the Checkpoint profile with synthetic parameters for a larger Qwen3 model. |
| `run.sh` | Runs every session, then `summarize.py`, and prints the numbers. |
| `summarize.py` | Turns the sessions under `paper/runs/cost` into `results.json`. |
| `results.json` | The numbers of Figure 7 and Section 6.5, from the paper's runs. |

## Commands

Run these from the repository root.

```bash
bash paper/cost/run.sh              # every session, then results.json and its numbers
python paper/cost/summarize.py      # results.json from the sessions under paper/runs/cost
python paper/show.py cost           # the numbers of Figure 7 and Section 6.5
```

Each arm generates exactly 256 tokens per request with greedy decoding.
It runs one warm-up and five measured repetitions per round.
LMSM arms evaluate their policy at each declared point without acting, which keeps the work identical across runs.
A ratio divides the median of an arm's ten measured repetitions by the median of native vLLM in the same session.

## GPU time

These are estimates for one NVIDIA H100 GPU, from the time each arm took in the paper's sessions plus the idle waits.

| Session | Arms | Widths | Time |
|---|---|---|---|
| `defaults` | native vLLM and each backend, with vLLM's default optimizations | 1, 32, 128 | 46 min |
| `matched` | native vLLM and each backend under matched settings | 1, 32, 128 | 48 min |
| `alternatives_probes` | native vLLM with its defaults and under matched settings, and the Python hook | 1, 32, 128 | 26 min |
| `alternatives_transcoder` | native vLLM with its defaults, and host evaluation of the transcoder | 1, 32, 128 | 18 min |
| `alternatives_sae` | native vLLM with its defaults on Gemma-3-4B, and host evaluation of the SAE | 1, 32, 128 | 21 min |
| `size_4b` to `size_32b` | native vLLM and the synthetic Checkpoint policy | 32, 128 | 8, 10, 13 and 22 min |

## Where each number comes from

All keys are in `results.json`. Widths are listed as 1, 32 and 128 unless noted.

| Paper | Key |
|---|---|
| Figure 7a, the four backends and every repetition | `defaults.<backend>.pct`, `runs_pct` and `native_runs_pct` |
| Figure 7b, Checkpoint at widths 32 and 128 | `model_size.<size>.pct` and `runs_pct` |
| Figure 7c, the band | the lowest and highest `defaults.<backend>.pct` at each width |
| Figure 7c, the four alternatives | `alternatives.matched_settings`, `python_hook`, `transcoder_host` and `sae_host` |
| "at least 96.4% of native throughput" (Section 6.5 and the Figure 7 caption) | `text.defaults_lowest_pct` |
| "the probe policies keep 97.9 to 99.5%" under matched settings | `text.matched_probes_pct` |
| "Matched settings ... give up 15 to 19%" | `text.matched_settings_loss_pct` |
| "the transcoder's features on the host then keeps only 22% at width 128" | `text.transcoder_host_width128_pct` |
| "the SAE's only 17%" | `text.sae_host_width128_pct` |
| "a Python forward hook ... keeps only 35 to 49%" | `text.python_hook_pct` |
| "98.6 to 99.4% of native throughput at width 32" on Qwen3-8B, 14B and 32B | `text.larger_models_width32_pct` |
| "compared with 96.4% on Qwen3-4B" | `text.qwen3_4b_width32_pct` |
| "within 0.1 points of the fitted policy's" (the appendix) | `text.synthetic_minus_fitted_points` |
