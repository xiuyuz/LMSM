"""In-process vLLM 0.19 integration for profile-selected backends.

The integration supports the evaluated offline path: one worker,
one activation hook, no speculative decoding, and buffered output.  vLLM's
``BatchUpdate`` is the only source of request identity.  The model-runner input
metadata supplies the current packed-token layout for each forward pass.  The
hook site and hidden width come from the active backend binding and loaded
model configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Callable

import torch

from lmsm.state import Decision


REQUEST_ID_EXTRA_ARG = "lmsm_request_id"


@dataclass
class ActionRecord:
    request_id: str
    step: int
    action: str
    rule_id: str | None
    target_id: str | None
    score: float | None
    threshold: float | None
    policy_name: str
    policy_version: str
    rule_library_version: str
    active_rule_ids: list[str]
    ordinary_model_prefix_tokens: list[int]
    forced_termination_token: int
    fixed_refusal_tokens: list[int]
    admitted_prefix_tokens: list[int]
    released_output_tokens: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequestState:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    step: int = 0
    actioned: bool = False


class BatchPolicyState:
    """Worker-local join between vLLM batch updates, hook rows, and logits."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rows: dict[int, RequestState] = {}
        self.evaluator: Any | None = None
        self.fixed_refusal_tokens: list[int] = []
        self.eos_token_id: int | None = None
        self.forward_epoch = 0
        self.captured_epoch = 0
        self.consumed_epoch = 0
        self.current_rows: list[int] = []
        self.final_token_indices: list[int] = []
        self.pending_actions: dict[int, Decision] = {}
        self.action_records: list[ActionRecord] = []
        self.archived_feature_maxima: dict[str, dict[str, dict[int, float]]] = {}
        self.archived_policy_scores: dict[str, dict[str, Any]] = {}
        self.archived_policy_score_steps: dict[str, dict[str, int]] = {}
        self.archived_policy_actions: dict[str, dict[str, bool]] = {}
        self.skip_capture = False
        self.activation_key: str | None = None
        self.expected_hidden_width: int | None = None
        self.hook_name: str | None = None

    def configure(self, evaluator: Any, fixed_refusal_tokens: list[int]) -> None:
        self.reset()
        self.evaluator = evaluator
        self.fixed_refusal_tokens = list(fixed_refusal_tokens)
        binding_ids = {rule.binding_id for rule in evaluator.profile.active_rules}
        if len(binding_ids) != 1:
            raise ValueError("LMSM vLLM requires one active backend binding")
        binding = evaluator.profile.binding_for(next(iter(binding_ids)))
        self.activation_key = binding.activation_key
        configured_width = binding.config.get("hidden_width")
        if configured_width is not None:
            self.expected_hidden_width = int(configured_width)

    def update_batch(self, batch_update: Any | None) -> None:
        if batch_update is None:
            return

        for row in batch_update.removed:
            removed = self.rows.pop(int(row), None)
            if removed is not None:
                self._archive_and_remove(removed.request_id)

        for row, params, prompt_ids, output_ids in batch_update.added:
            row = int(row)
            extra_args = params.extra_args or {}
            request_id = extra_args.get(REQUEST_ID_EXTRA_ARG)
            if request_id is None or not str(request_id):
                raise ValueError(
                    f"SamplingParams.extra_args must contain {REQUEST_ID_EXTRA_ARG!r}"
                )
            previous = self.rows.get(row)
            if previous is not None:
                self._archive_and_remove(previous.request_id)
            state = RequestState(
                request_id=str(request_id),
                prompt_token_ids=list(prompt_ids or []),
                output_token_ids=output_ids,
            )
            self.rows[row] = state
            self.archived_feature_maxima.pop(state.request_id, None)
            self.archived_policy_scores.pop(state.request_id, None)
            self.archived_policy_score_steps.pop(state.request_id, None)
            self.archived_policy_actions.pop(state.request_id, None)
            self.evaluator.add(state.request_id)

        for source, destination, directionality in batch_update.moved:
            source, destination = int(source), int(destination)
            if directionality.name == "SWAP":
                source_state = self.rows.pop(source, None)
                destination_state = self.rows.pop(destination, None)
                if source_state is not None:
                    self.rows[destination] = source_state
                if destination_state is not None:
                    self.rows[source] = destination_state
            else:
                state = self.rows.pop(source, None)
                if state is not None:
                    self.rows[destination] = state

        for row in list(self.rows):
            if row >= int(batch_update.batch_size):
                removed = self.rows.pop(row)
                self._archive_and_remove(removed.request_id)

    def _archive_and_remove(self, request_id: str) -> None:
        self.archived_feature_maxima[request_id] = self.evaluator.feature_maxima_for(
            request_id
        )
        scores_for = getattr(self.evaluator, "scores_for", None)
        score_steps_for = getattr(self.evaluator, "score_steps_for", None)
        actions_for = getattr(self.evaluator, "actions_for", None)
        self.archived_policy_scores[request_id] = (
            scores_for(request_id) if scores_for is not None else {}
        )
        self.archived_policy_score_steps[request_id] = (
            score_steps_for(request_id) if score_steps_for is not None else {}
        )
        self.archived_policy_actions[request_id] = (
            actions_for(request_id) if actions_for is not None else {}
        )
        self.evaluator.remove(request_id)

    def feature_maxima_for(self, request_id: str) -> dict[str, dict[int, float]]:
        if request_id in self.archived_feature_maxima:
            return self.archived_feature_maxima[request_id]
        return self.evaluator.feature_maxima_for(request_id)

    def policy_scores_for(self, request_id: str) -> dict[str, Any]:
        if request_id in self.archived_policy_scores:
            return dict(self.archived_policy_scores[request_id])
        scores_for = getattr(self.evaluator, "scores_for", None)
        return scores_for(request_id) if scores_for is not None else {}

    def policy_actions_for(self, request_id: str) -> dict[str, bool]:
        if request_id in self.archived_policy_actions:
            return dict(self.archived_policy_actions[request_id])
        actions_for = getattr(self.evaluator, "actions_for", None)
        return actions_for(request_id) if actions_for is not None else {}

    def policy_score_steps_for(self, request_id: str) -> dict[str, int]:
        if request_id in self.archived_policy_score_steps:
            return dict(self.archived_policy_score_steps[request_id])
        score_steps_for = getattr(self.evaluator, "score_steps_for", None)
        return score_steps_for(request_id) if score_steps_for is not None else {}

    def begin_forward(self, num_scheduled_tokens: Any) -> None:
        counts = [int(value) for value in num_scheduled_tokens]
        if any(count <= 0 for count in counts):
            raise AssertionError("each active vLLM request must schedule at least one token")
        current_rows = list(range(len(counts)))
        if any(row not in self.rows for row in current_rows):
            raise AssertionError("current post-update batch has an unmapped request row")

        running = 0
        final_indices = []
        for count in counts:
            running += count
            final_indices.append(running - 1)

        eligible_rows = []
        eligible_indices = []
        for row, count, final_index in zip(
            current_rows, counts, final_indices, strict=True
        ):
            request = self.rows[row]
            step = len(request.output_token_ids)
            prompt_endpoint = (
                self.evaluator.observes_prompt
                and step == 0
                and count == len(request.prompt_token_ids)
                and self.evaluator.needs_prompt(request.request_id)
            )
            max_decode_step = self.evaluator.max_decode_step
            decode_endpoint = (
                self.evaluator.observes_decode
                and step > 0
                and not request.actioned
                and (max_decode_step is None or step <= max_decode_step)
            )
            if prompt_endpoint or decode_endpoint:
                eligible_rows.append(row)
                eligible_indices.append(final_index)
        self.skip_capture = not eligible_rows
        current_rows = eligible_rows
        final_indices = eligible_indices
        self.forward_epoch += 1
        self.current_rows = current_rows
        self.final_token_indices = final_indices
        self.pending_actions = {}

    def capture(self, hidden: torch.Tensor) -> None:
        if self.skip_capture:
            if self.captured_epoch == self.forward_epoch:
                raise AssertionError("duplicate activation capture in one forward")
            self.pending_actions = {}
            self.captured_epoch = self.forward_epoch
            return
        live_forward = self.forward_epoch > self.captured_epoch
        if not self.rows and not live_forward:
            return
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("profile-selected activation was not a tensor")
        if self.expected_hidden_width is None:
            self.expected_hidden_width = int(hidden.shape[-1])
        if hidden.shape[-1] != self.expected_hidden_width:
            raise AssertionError(
                f"expected hidden width {self.expected_hidden_width}, "
                f"got {hidden.shape[-1]}"
            )
        if not self.final_token_indices:
            raise AssertionError("hook fired without current forward metadata")
        if self.captured_epoch == self.forward_epoch:
            raise AssertionError("duplicate activation capture in one forward")

        final_rows = hidden[self.final_token_indices]
        request_ids = [self.rows[row].request_id for row in self.current_rows]
        steps = [len(self.rows[row].output_token_ids) for row in self.current_rows]
        decisions = self.evaluator.evaluate(request_ids, final_rows, steps)
        if len(decisions) != len(self.current_rows):
            raise AssertionError("policy evaluator returned the wrong number of decisions")
        self.pending_actions = {
            row: decision
            for row, decision in zip(self.current_rows, decisions, strict=True)
            if decision.action != "allow" and not self.rows[row].actioned
        }
        self.captured_epoch = self.forward_epoch

    def apply_actions(self, logits: torch.Tensor) -> torch.Tensor:
        if self.captured_epoch != self.forward_epoch:
            raise AssertionError("logits processor did not receive current-step hook data")
        if self.consumed_epoch == self.forward_epoch or not self.pending_actions:
            self.consumed_epoch = self.forward_epoch
            return logits
        if self.eos_token_id is None:
            raise AssertionError("termination token is not configured")

        for row, decision in self.pending_actions.items():
            request = self.rows[row]
            prefix = list(request.output_token_ids)
            profile = self.evaluator.profile
            released = list(self.fixed_refusal_tokens) if decision.action == "refuse" else []
            self.action_records.append(
                ActionRecord(
                    request_id=request.request_id,
                    step=len(prefix),
                    action=decision.action,
                    rule_id=decision.params.get("rule_id"),
                    target_id=decision.params.get("target_id"),
                    score=decision.params.get("score"),
                    threshold=decision.params.get("threshold"),
                    policy_name=profile.policy.name,
                    policy_version=profile.policy.version,
                    rule_library_version=profile.rule_library.version,
                    active_rule_ids=list(profile.policy.active_rule_ids),
                    ordinary_model_prefix_tokens=prefix,
                    forced_termination_token=self.eos_token_id,
                    fixed_refusal_tokens=list(self.fixed_refusal_tokens),
                    admitted_prefix_tokens=prefix,
                    released_output_tokens=released,
                )
            )
            request.actioned = True
            request.step = len(prefix)
            logits[row].fill_(float("-inf"))
            logits[row, self.eos_token_id] = 0.0

        self.pending_actions = {}
        self.consumed_epoch = self.forward_epoch
        return logits

    def record_for(self, request_id: str) -> ActionRecord | None:
        return next(
            (
                record
                for record in reversed(self.action_records)
                if record.request_id == request_id
            ),
            None,
        )


STATE = BatchPolicyState()


try:
    from vllm.v1.sample.logits_processor import LogitsProcessor
except ImportError:  # Allows CPU unit tests without importing a CUDA runtime.
    class LogitsProcessor:  # type: ignore[no-redef]
        pass


class VLLMEnforcementGate(LogitsProcessor):
    """Force EOS on current rows selected by the configured worker hook."""

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        extra_args = sampling_params.extra_args or {}
        if not extra_args.get(REQUEST_ID_EXTRA_ARG):
            raise ValueError(f"missing stable {REQUEST_ID_EXTRA_ARG}")
        temperature = float(sampling_params.temperature)
        top_p = float(sampling_params.top_p)
        top_k = int(sampling_params.top_k)
        greedy = temperature == 0.0
        nucleus = temperature > 0.0 and 0.0 < top_p < 1.0 and top_k >= -1
        if not (greedy or nucleus):
            raise ValueError("LMSM vLLM supports only greedy or nucleus sampling")

    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool):
        del device, is_pin_memory
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("LMSM vLLM requires TP=1 and PP=1")
        if vllm_config.speculative_config is not None:
            raise ValueError("LMSM vLLM does not support speculative decoding")
        eos = vllm_config.model_config.hf_config.eos_token_id
        STATE.eos_token_id = int(eos[0] if isinstance(eos, list) else eos)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Any | None) -> None:
        STATE.update_batch(batch_update)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return STATE.apply_actions(logits)


class MatchedEmptyExtensionProcessor(LogitsProcessor):
    """Loaded vLLM extension with no backend, policy, or request state."""

    @classmethod
    def validate_params(cls, _sampling_params: Any) -> None:
        return None

    def __init__(self, _vllm_config: Any, _device: torch.device, _is_pin_memory: bool):
        pass

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, _batch_update: Any | None) -> None:
        return None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


_PATCHED = False
_ORIGINAL_LOAD_MODEL: Callable[..., Any] | None = None
_ORIGINAL_PREPARE_INPUTS: Callable[..., Any] | None = None


def install_vllm_integration(evaluator: Any, fixed_refusal_tokens: list[int]) -> None:
    """Install the profile-selected hook and current-input metadata bridge."""

    global _PATCHED, _ORIGINAL_LOAD_MODEL, _ORIGINAL_PREPARE_INPUTS
    if _PATCHED:
        raise RuntimeError("LMSM vLLM runtime is already installed")
    STATE.configure(evaluator, fixed_refusal_tokens)

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _ORIGINAL_LOAD_MODEL = GPUModelRunner.load_model
    _ORIGINAL_PREPARE_INPUTS = GPUModelRunner._prepare_inputs

    @wraps(_ORIGINAL_LOAD_MODEL)
    def load_model_and_hook(runner: Any, *args: Any, **kwargs: Any) -> None:
        assert _ORIGINAL_LOAD_MODEL is not None
        _ORIGINAL_LOAD_MODEL(runner, *args, **kwargs)
        activation_key = STATE.activation_key
        if activation_key is None:
            raise AssertionError("LMSM activation key is not configured")
        hook_mode, module_name = _split_hook_target(activation_key)
        matches = [
            (name, module)
            for name, module in runner.model.named_modules()
            if name == module_name
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected exactly one {module_name!r}, found {len(matches)}"
            )
        name, module = matches[0]
        STATE.hook_name = name
        if hook_mode == "pre":
            module.register_forward_pre_hook(_capture_module_input)
        else:
            module.register_forward_hook(_capture_module_output)

        model_width = _model_hidden_width(runner.model_config.hf_config)
        if STATE.expected_hidden_width is None:
            STATE.expected_hidden_width = model_width

    @wraps(_ORIGINAL_PREPARE_INPUTS)
    def prepare_inputs(runner: Any, scheduler_output: Any, num_scheduled_tokens: Any):
        assert _ORIGINAL_PREPARE_INPUTS is not None
        prepared = _ORIGINAL_PREPARE_INPUTS(
            runner, scheduler_output, num_scheduled_tokens
        )
        # This array is in the current post-update InputBatch row order.  Its
        # values define the packed forward layout; no scheduler dictionary or
        # internal request ID participates in the join.
        STATE.begin_forward(num_scheduled_tokens)
        return prepared

    GPUModelRunner.load_model = load_model_and_hook
    GPUModelRunner._prepare_inputs = prepare_inputs
    _PATCHED = True


def uninstall_vllm_integration() -> None:
    global _PATCHED, _ORIGINAL_LOAD_MODEL, _ORIGINAL_PREPARE_INPUTS
    if not _PATCHED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    GPUModelRunner.load_model = _ORIGINAL_LOAD_MODEL
    GPUModelRunner._prepare_inputs = _ORIGINAL_PREPARE_INPUTS
    _ORIGINAL_LOAD_MODEL = None
    _ORIGINAL_PREPARE_INPUTS = None
    _PATCHED = False


def _capture_module_input(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
    STATE.capture(args[0])


def _capture_module_output(
    _module: torch.nn.Module, _args: tuple[Any, ...], output: Any
) -> None:
    STATE.capture(output[0] if isinstance(output, tuple) else output)


def _split_hook_target(activation_key: str) -> tuple[str, str]:
    if activation_key.startswith("pre:"):
        return "pre", activation_key[len("pre:") :]
    return "post", activation_key


def _model_hidden_width(hf_config: Any) -> int | None:
    hidden_width = getattr(hf_config, "hidden_size", None)
    if hidden_width is None:
        text_config = getattr(hf_config, "text_config", None)
        hidden_width = getattr(text_config, "hidden_size", None)
    return int(hidden_width) if hidden_width is not None else None


def attach_request_id(sampling_params: Any, request_id: str) -> Any:
    """Attach the stable external request ID used by the worker-local join."""

    sampling_params.extra_args = dict(sampling_params.extra_args or {})
    sampling_params.extra_args[REQUEST_ID_EXTRA_ARG] = str(request_id)
    return sampling_params


def buffered_output(request_id: str, ordinary_output_tokens: list[int]) -> dict[str, Any]:
    """Return the compact offline output view without exposing forced EOS."""

    record = STATE.record_for(request_id)
    if record is None:
        return {
            "request_id": request_id,
            "ordinary_model_prefix_tokens": list(ordinary_output_tokens),
            "forced_termination_token": None,
            "fixed_refusal_tokens": [],
            "admitted_prefix_tokens": list(ordinary_output_tokens),
            "released_output_tokens": list(ordinary_output_tokens),
            "action_record": None,
        }
    return {**record.to_dict(), "action_record": record.to_dict()}


def feature_maxima(request_id: str) -> dict[str, dict[int, float]]:
    """Return compact max-over-forward raw scores for one completed request."""

    return STATE.feature_maxima_for(request_id)


def policy_scores(request_id: str) -> dict[str, Any]:
    """Return the evaluator's saved rule scores for one request."""

    return STATE.policy_scores_for(request_id)


def policy_actions(request_id: str) -> dict[str, bool]:
    """Return the evaluator's per-rule threshold outcomes for one request."""

    return STATE.policy_actions_for(request_id)


def policy_score_steps(request_id: str) -> dict[str, int]:
    """Return the decode step associated with each saved rule score."""

    return STATE.policy_score_steps_for(request_id)


def backend_evaluation_count() -> int | None:
    """Return the shared feature backend's encode count when it exposes one."""

    shared_backend = getattr(STATE.evaluator, "shared_backend", None)
    return getattr(shared_backend, "encode_count", None)
