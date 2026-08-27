# LMSM

LMSM is a generation-time safety substrate for language models. It connects
internal model signals to small, explicit safety rules, keeps policy state per
request, and mediates only the requests whose active rules trigger.

For the complete design and evaluation, see
[LMSM: LLM Security Framework Inspired by Linux Security Modules](http://arxiv.org/abs/2608.25697).

The core control path is:

```text
Targets -> Backend bindings -> Rules -> Policy bundle
                                      |
model forward -> request-keyed backend evaluation -> composition -> action -> response
```

## Control-plane concepts

- A **Target** names one behavior that a rule can detect, such as cybercrime or
  harassment, together with the scope in which the rule applies.
- A **Backend binding** connects named signal channels to an activation site
  and a fitted dense probe, SAE, or transcoder backend.
- A **Rule** binds one target to backend channels, a calibrated condition, and
  a candidate action. Rules are independent and reusable.
- A **Policy bundle** selects active rules, their temporal schedule, fixed OR
  composition order, and the mapping from candidate actions to runtime actions.

During generation, LMSM associates every evaluated row with its request ID. Each
request therefore has its own running scores, threshold state, and action
history even when vLLM packs many requests into one forward pass. The runtime
applies an action only to selected rows; other rows continue normally.

## Included profiles

The portable YAML files under `profiles/` define four ready-to-use configurations:

| Profile | Schedule |
|---|---|
| `LMSM-Checkpoint` | Pools the prompt and two generated-token windows, then evaluates their calibrated fusion at step 64. |
| `LMSM-Anytime` | Updates running-prefix scores at every generated step and acts at the first crossing. |
| `LMSM-Transcoder` | Tracks calibrated Qwen3 transcoder features during the prompt and generation, then acts at the first crossing. |
| `LMSM-SAE` | Tracks calibrated Gemma Scope SAE features during the prompt and generation, then acts at the first crossing. |

Checkpoint and Anytime use standardized dense linear probes.
Transcoder and SAE provide complete artifact-backed examples of the other two
backend types: each contains six independent semantic rules with its calibrated
feature coordinates and thresholds. A policy can activate the full rule set or
any subset without changing the backend definition.

Each profile keeps target order, backend details, rule selection, scheduling,
composition, and mapping to the public `allow`, `refuse`, and `terminate`
actions in one `DeploymentProfile` object. Bindings, individual rules, rule
libraries, and policy bundles use simple human-readable version strings.

The included builders implement the schedules above. A custom policy
evaluator can implement additional schedule semantics against the same control
plane and enforcement interfaces.

## Installation

LMSM requires Python 3.10 or newer. Install the base library and the optional
runtime you intend to use:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[hf]'
```

For the offline vLLM integration:

```bash
python -m pip install -e '.[hf,vllm]'
```

Install the SAE extra to use `LMSM-SAE`:

```bash
python -m pip install -e '.[hf,sae]'
```

The included in-process vLLM adapter targets vLLM 0.19.0 with one worker and
tensor and pipeline parallel size 1. It reads the activation module, pre/post
hook mode, and hidden width from the active backend binding and loaded model
rather than assuming a particular model architecture. The Hugging Face
integration uses the same profile-selected activation sites.

## Hugging Face usage

```python
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from lmsm import build_transformers_engine, load_profile

root = Path("/path/to/LMSM")
model_id = "Qwen/Qwen3-4B"
profile = load_profile(root / "profiles/lmsm_anytime.yaml")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

engine = build_transformers_engine(model, tokenizer, profile, root)
print(engine.generate("Explain how photosynthesis works."))
```

Select independent rules before building the engine when only part of a policy
is needed:

```python
profile = profile.with_active_rules([
    "chemical_biological",
    "cybercrime_intrusion",
])
```

Set `actions_enabled=False` in `build_transformers_engine` to evaluate the
policy without changing responses.

## vLLM usage

`scripts/run_vllm.py` accepts JSONL rows with `id` and `prompt` fields and
writes `outputs.jsonl`, `actions.jsonl`, and `timings.csv`:

```bash
python scripts/run_vllm.py \
  --model Qwen/Qwen3-4B \
  --profile profiles/lmsm_anytime.yaml \
  --input examples/requests.jsonl \
  --run-dir runs/anytime \
  --physical-gpu 0
```

Repeat `--active-rule RULE_ID` to run a selected rule subset. For example,
`--active-rule chemical_biological --active-rule cybercrime_intrusion` activates
only those two rules from the chosen profile.

## Tests

The repository includes the fast CPU test suite used during development:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

These tests cover the library interfaces and the supplied profile contracts.
GPU benchmark reproduction is documented separately in `reproduction/`.

## License

LMSM source code and project-authored documentation use the MIT License.
Models and backend artifacts retain their own licenses.
