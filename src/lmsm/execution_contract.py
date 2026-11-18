"""LMSM execution contract: declared needs, per-request accounting, and release.

A policy declares which processed positions it must observe and where it
evaluates (``ExecutionRequirements``).  The serving runtime checks the engine
configuration once at startup (``check_engine_arguments`` and
``check_vllm_config``).  It keeps one ``RequestAccount`` per trusted request ID
and gives every finished request exactly one state, checked in this order.

``execution_failed``
    A position the request processed, or an evaluation point it reached, lacks
    its observation or evaluation.  A request also fails when the runtime
    detects a fault, such as an evaluator error, a policy-state reset, or a NaN
    or infinite activation or score that the policy uses at a processed step.
    Under asynchronous scheduling the engine may discard the token of a forward
    at or past the final length.  A non-finite value seen only on such a
    forward is recorded as ``discarded_nonfinite_step`` and is not a failure.
``not_reached``
    Every processed position that required an observation was observed, but
    the request finished before its first evaluation point.
``completed``
    All required observations are present and every reached evaluation point
    was evaluated.

A forward pass at step ``s`` processes the prompt (``s == 0``) or generated
token ``s`` (``s >= 1``).  A request that generated ``N`` tokens therefore
processed steps ``0 .. N - 1``.  The final generated token is never processed,
so no policy requires its observation.

The release step uses the state (``release_outcome``).  A completed request is
released as its policy decided.  A not-reached request follows the policy's
declared disposition.  A failed request receives the fixed refusal with a
recorded reason, never a silent allow.  The contract detects missing evidence.
It does not detect evidence attributed to the wrong request, which request-ID
keying prevents.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


EXECUTION_FAILED = "execution_failed"
NOT_REACHED = "not_reached"
COMPLETED = "completed"
EXECUTION_STATES = (EXECUTION_FAILED, NOT_REACHED, COMPLETED)

ALLOW_WITH_RECORD = "allow_with_record"
DENY = "deny"
NOT_REACHED_DISPOSITIONS = (ALLOW_WITH_RECORD, DENY)

RELEASE_ORDINARY = "ordinary"
RELEASE_REFUSAL = "refusal"
RELEASE_NONE = "none"

SUPPORTED_VLLM_SERIES = "0.19."


def validate_not_reached_disposition(value: str) -> str:
    value = str(value)
    if value not in NOT_REACHED_DISPOSITIONS:
        raise ValueError(
            f"not-reached disposition must be one of {NOT_REACHED_DISPOSITIONS}, "
            f"got {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# Policy declarations


@dataclass(frozen=True)
class ObservationWindow:
    """A contiguous range of processed steps whose observations are required."""

    name: str
    first_step: int
    last_step: int | None = None  # None: open-ended (until the decision)

    def contains(self, step: int) -> bool:
        return step >= self.first_step and (
            self.last_step is None or step <= self.last_step
        )


@dataclass(frozen=True)
class ExecutionRequirements:
    """What one policy needs from the runtime in order to count as executed.

    ``windows`` lists the processed steps that must be observed.  When
    ``evaluation_steps`` is ``None`` every required observation is also an
    evaluation point (running schedules).  Otherwise only the listed steps are.
    Once the runtime applies a non-allow decision at step ``d``, later steps
    are neither required nor evaluation points.
    """

    policy_name: str
    policy_version: str
    rule_library_version: str
    schedule: dict[str, Any]
    windows: tuple[ObservationWindow, ...]
    evaluation_steps: tuple[int, ...] | None = None
    not_reached_disposition: str = ALLOW_WITH_RECORD
    coverage: str = ""

    def __post_init__(self) -> None:
        validate_not_reached_disposition(self.not_reached_disposition)

    @classmethod
    def for_profile(
        cls,
        profile: Any,
        *,
        windows: Iterable[ObservationWindow],
        evaluation_steps: Iterable[int] | None = None,
        coverage: str = "",
    ) -> "ExecutionRequirements":
        policy = profile.policy
        return cls(
            policy_name=str(policy.name),
            policy_version=str(policy.version),
            rule_library_version=str(profile.rule_library.version),
            schedule=_schedule_dict(policy.schedule),
            windows=tuple(windows),
            evaluation_steps=(
                tuple(int(step) for step in evaluation_steps)
                if evaluation_steps is not None
                else None
            ),
            not_reached_disposition=str(
                getattr(policy, "not_reached_disposition", ALLOW_WITH_RECORD)
            ),
            coverage=coverage,
        )

    def requires_observation(self, step: int, decision_step: int | None) -> bool:
        if decision_step is not None and step > decision_step:
            return False
        return any(window.contains(step) for window in self.windows)

    def is_evaluation_point(self, step: int, decision_step: int | None) -> bool:
        if decision_step is not None and step > decision_step:
            return False
        if self.evaluation_steps is None:
            return self.requires_observation(step, decision_step)
        return step in self.evaluation_steps

    def first_evaluation_step(self) -> int | None:
        if self.evaluation_steps is not None:
            return min(self.evaluation_steps) if self.evaluation_steps else None
        return min((window.first_step for window in self.windows), default=None)

    def evaluation_points_description(self) -> dict[str, Any]:
        if self.evaluation_steps is not None:
            return {"kind": "fixed_steps", "steps": list(self.evaluation_steps)}
        return {
            "kind": "every_required_observation",
            "first_step": self.first_evaluation_step(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "rule_library_version": self.rule_library_version,
            "schedule": dict(self.schedule),
            "required_observation_windows": [asdict(window) for window in self.windows],
            "evaluation_points": self.evaluation_points_description(),
            "not_reached_disposition": self.not_reached_disposition,
            "execution_failed_response": "fixed_refusal",
            "coverage": self.coverage,
            "processed_token_semantics": (
                "step 0 processes the prompt and step s >= 1 processes generated "
                "token s; the final generated token is never processed"
            ),
        }


def derive_execution_requirements(evaluator: Any) -> ExecutionRequirements:
    """Return an evaluator's declaration, deriving one for custom evaluators.

    Built-in evaluators implement ``execution_requirements()``.  A custom
    evaluator without it is treated as a running schedule over the positions
    its ``observes_prompt``, ``observes_decode``, and ``max_decode_step``
    attributes select, evaluated at every observation.
    """

    declared = getattr(evaluator, "execution_requirements", None)
    if callable(declared):
        return declared()
    windows = []
    if getattr(evaluator, "observes_prompt", False):
        windows.append(ObservationWindow("prompt", 0, 0))
    if getattr(evaluator, "observes_decode", False):
        max_decode_step = getattr(evaluator, "max_decode_step", None)
        windows.append(
            ObservationWindow(
                "decode",
                1,
                int(max_decode_step) if max_decode_step is not None else None,
            )
        )
    return ExecutionRequirements.for_profile(
        evaluator.profile,
        windows=windows,
        evaluation_steps=None,
        coverage="derived from the evaluator's observation attributes",
    )


def _schedule_dict(schedule: Any) -> dict[str, Any]:
    keys = (
        "kind",
        "pooling",
        "evaluation_step",
        "window_split_step",
        "max_decode_step",
        "first_crossing",
    )
    return {key: getattr(schedule, key, None) for key in keys}


# ---------------------------------------------------------------------------
# Per-request accounting


class _StepSet:
    """A compact set of non-negative step indices."""

    __slots__ = ("_bits",)

    def __init__(self) -> None:
        self._bits = bytearray()

    def add(self, step: int) -> None:
        if step < 0:
            raise ValueError("steps are non-negative")
        if step >= len(self._bits):
            self._bits.extend(b"\x00" * (step + 1 - len(self._bits)))
        self._bits[step] = 1

    def discard(self, step: int) -> None:
        if 0 <= step < len(self._bits):
            self._bits[step] = 0

    def __contains__(self, step: int) -> bool:
        return 0 <= step < len(self._bits) and self._bits[step] == 1

    def count_in(self, steps: Iterable[int]) -> int:
        return sum(1 for step in steps if step in self)


@dataclass
class RuntimeFailure:
    step: int | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "reason": self.reason}


class RequestAccount:
    """Observation and evaluation account for one trusted request ID."""

    __slots__ = (
        "request_id",
        "prompt_length",
        "admissions",
        "processed",
        "observed",
        "evaluated",
        "decision_step",
        "decision",
        "failures",
        "injected_faults",
        "state_key",
        "foreign_evidence",
        "decision_used_foreign_evidence",
        "discarded_nonfinite_step",
    )

    def __init__(self, request_id: str, prompt_length: int) -> None:
        self.request_id = str(request_id)
        self.prompt_length = int(prompt_length)
        self.admissions = 1
        self.processed = _StepSet()
        self.observed = _StepSet()
        self.evaluated = _StepSet()
        self.decision_step: int | None = None
        self.decision: str | None = None
        self.failures: list[RuntimeFailure] = []
        self.injected_faults: list[dict[str, Any]] = []
        # Bookkeeping for the request-association ablation (mediation tests only).
        self.state_key: str | None = None
        self.foreign_evidence: set[str] = set()
        self.decision_used_foreign_evidence: bool | None = None
        # Non-finite evidence seen only on a forward whose sampled token the
        # engine discarded (async scheduling).  A diagnostic, not a failure.
        self.discarded_nonfinite_step: int | None = None

    @property
    def failed(self) -> bool:
        return bool(self.failures)

    def fail(self, step: int | None, reason: str) -> None:
        self.failures.append(RuntimeFailure(step, reason))

    def mark_processed(self, step: int) -> None:
        self.processed.add(step)

    def mark_evaluated(self, step: int) -> None:
        self.observed.add(step)
        self.evaluated.add(step)

    def record_decision(self, step: int, action: str) -> None:
        if self.decision_step is None:
            self.decision_step = int(step)
            self.decision = str(action)


@dataclass
class ExecutionOutcome:
    state: str
    reason: str
    generated_token_count: int
    decision: str | None
    decision_step: int | None
    disposition: str | None
    release: str
    windows: list[dict[str, Any]] = field(default_factory=list)
    evaluation_points_reached: list[int] = field(default_factory=list)
    evaluation_points_evaluated: int = 0
    missing_observation_steps: list[int] = field(default_factory=list)
    missing_evaluation_steps: list[int] = field(default_factory=list)
    runtime_failures: list[dict[str, Any]] = field(default_factory=list)


def assign_execution_state(
    requirements: ExecutionRequirements,
    account: RequestAccount | None,
    generated_token_count: int,
) -> ExecutionOutcome:
    """Assign exactly one execution state and the release it implies.

    ``generated_token_count`` is the number of tokens the engine generated for
    the request.  The processed steps are taken from it (``0 .. N - 1``) rather
    than from the runtime's own instrumentation, so an instrumentation gap
    cannot make an unobserved position disappear from the account.
    """

    count = max(int(generated_token_count), 0)
    processed_steps = range(count)
    if account is None:
        required = [
            step for step in processed_steps
            if requirements.requires_observation(step, None)
        ]
        reached = [
            step for step in processed_steps
            if requirements.is_evaluation_point(step, None)
        ]
        return ExecutionOutcome(
            state=EXECUTION_FAILED,
            reason="no execution account: the runtime never admitted this request",
            generated_token_count=count,
            decision=None,
            decision_step=None,
            disposition=DENY,
            release=RELEASE_REFUSAL,
            windows=_window_counts(requirements, processed_steps, None, None),
            evaluation_points_reached=reached,
            missing_observation_steps=required,
            missing_evaluation_steps=reached,
        )

    decision_step = account.decision_step
    required = [
        step for step in processed_steps
        if requirements.requires_observation(step, decision_step)
    ]
    reached = [
        step for step in processed_steps
        if requirements.is_evaluation_point(step, decision_step)
    ]
    missing_observations = [step for step in required if step not in account.observed]
    missing_evaluations = [step for step in reached if step not in account.evaluated]
    failures = [failure.to_dict() for failure in account.failures]
    if decision_step is not None and decision_step >= count:
        failures.append(
            {
                "step": decision_step,
                "reason": "decision recorded for a position the request did not process",
            }
        )
    if decision_step is not None and not requirements.is_evaluation_point(
        decision_step, decision_step
    ):
        failures.append(
            {
                "step": decision_step,
                "reason": "decision recorded at a step the policy does not declare "
                "as an evaluation point",
            }
        )

    windows = _window_counts(requirements, processed_steps, decision_step, account)
    evaluated_count = sum(1 for step in reached if step in account.evaluated)
    if failures or missing_observations or missing_evaluations:
        parts = []
        if missing_observations:
            parts.append(
                f"missing observation at {len(missing_observations)} processed "
                f"step(s) {_step_summary(missing_observations)}"
            )
        if missing_evaluations:
            parts.append(
                f"missing evaluation at {len(missing_evaluations)} reached "
                f"evaluation point(s) {_step_summary(missing_evaluations)}"
            )
        seen = set()
        for failure in failures:
            if failure["reason"] not in seen:
                seen.add(failure["reason"])
                parts.append(failure["reason"])
        return ExecutionOutcome(
            state=EXECUTION_FAILED,
            reason="; ".join(parts),
            generated_token_count=count,
            decision=None,
            decision_step=decision_step,
            disposition=DENY,
            release=RELEASE_REFUSAL,
            windows=windows,
            evaluation_points_reached=reached,
            evaluation_points_evaluated=evaluated_count,
            missing_observation_steps=missing_observations,
            missing_evaluation_steps=missing_evaluations,
            runtime_failures=failures,
        )

    if not reached:
        disposition = requirements.not_reached_disposition
        first = requirements.first_evaluation_step()
        return ExecutionOutcome(
            state=NOT_REACHED,
            reason=(
                f"finished after {count} generated token(s), before the first "
                f"evaluation point (step {first}); all {len(required)} required "
                f"observation(s) present; disposition {disposition}"
            ),
            generated_token_count=count,
            decision=None,
            decision_step=None,
            disposition=disposition,
            release=RELEASE_ORDINARY if disposition == ALLOW_WITH_RECORD else RELEASE_REFUSAL,
            windows=windows,
            evaluation_points_reached=[],
            evaluation_points_evaluated=0,
        )

    decision = account.decision or "allow"
    if decision == "allow":
        release = RELEASE_ORDINARY
        reason = (
            f"all {len(required)} required observation(s) present and "
            f"{len(reached)} evaluation point(s) evaluated; decision allow"
        )
    else:
        release = RELEASE_REFUSAL if decision == "refuse" else RELEASE_NONE
        reason = (
            f"all {len(required)} required observation(s) present; decision "
            f"{decision} at step {decision_step}"
        )
    return ExecutionOutcome(
        state=COMPLETED,
        reason=reason,
        generated_token_count=count,
        decision=decision,
        decision_step=decision_step,
        disposition=None,
        release=release,
        windows=windows,
        evaluation_points_reached=reached,
        evaluation_points_evaluated=evaluated_count,
    )


def release_outcome(state: str, decision: str | None, disposition: str | None) -> str:
    """Map an execution state to the kind of response released."""

    if state == COMPLETED:
        if decision in (None, "allow"):
            return RELEASE_ORDINARY
        if decision == "refuse":
            return RELEASE_REFUSAL
        if decision == "terminate":
            return RELEASE_NONE
        raise KeyError(f"unknown decision: {decision}")
    if state == NOT_REACHED:
        validate_not_reached_disposition(disposition or ALLOW_WITH_RECORD)
        return RELEASE_ORDINARY if disposition in (None, ALLOW_WITH_RECORD) else RELEASE_REFUSAL
    if state == EXECUTION_FAILED:
        return RELEASE_REFUSAL
    raise KeyError(f"unknown execution state: {state}")


def build_execution_record(
    requirements: ExecutionRequirements,
    account: RequestAccount | None,
    outcome: ExecutionOutcome,
    *,
    request_id: str,
    finish_reason: str | None = None,
    association_ablation: bool = False,
) -> dict[str, Any]:
    """Build the per-request execution record (every request gets one)."""

    reached = outcome.evaluation_points_reached
    record: dict[str, Any] = {
        "request_id": str(request_id),
        "policy_name": requirements.policy_name,
        "policy_version": requirements.policy_version,
        "rule_library_version": requirements.rule_library_version,
        "schedule": dict(requirements.schedule),
        "required_evaluation_points": requirements.evaluation_points_description(),
        "generated_token_count": outcome.generated_token_count,
        "processed_positions": outcome.generated_token_count,
        "finish_reason": finish_reason,
        "observed_positions": outcome.windows,
        "evaluation_points_reached": (
            list(reached)
            if requirements.evaluation_steps is not None
            else len(reached)
        ),
        "evaluation_points_evaluated": outcome.evaluation_points_evaluated,
        "missing_observation_steps": list(outcome.missing_observation_steps),
        "missing_evaluation_steps": list(outcome.missing_evaluation_steps),
        "state": outcome.state,
        "decision": outcome.decision,
        "decision_step": outcome.decision_step,
        "disposition": outcome.disposition,
        "reason": outcome.reason,
        "release_outcome": outcome.release,
        "admissions": account.admissions if account is not None else 0,
        "runtime_failures": list(outcome.runtime_failures),
        "injected_faults": list(account.injected_faults) if account is not None else [],
    }
    if account is not None and account.discarded_nonfinite_step is not None:
        # Present only for such requests.
        record["discarded_nonfinite_step"] = account.discarded_nonfinite_step
    if association_ablation and account is not None:
        record["association_ablation"] = {
            "state_key": account.state_key,
            "foreign_evidence_request_ids": sorted(account.foreign_evidence),
            "decision_used_foreign_evidence": account.decision_used_foreign_evidence,
        }
    return record


def _window_counts(
    requirements: ExecutionRequirements,
    processed_steps: range,
    decision_step: int | None,
    account: RequestAccount | None,
) -> list[dict[str, Any]]:
    windows = []
    for window in requirements.windows:
        steps = [
            step for step in processed_steps
            if window.contains(step)
            and (decision_step is None or step <= decision_step)
        ]
        windows.append(
            {
                "window": window.name,
                "first_step": window.first_step,
                "last_step": window.last_step,
                "required": len(steps),
                "observed": account.observed.count_in(steps) if account is not None else 0,
            }
        )
    return windows


def _step_summary(steps: list[int], limit: int = 8) -> str:
    if len(steps) <= limit:
        return str(steps)
    return f"[{', '.join(str(step) for step in steps[:limit])}, ...]"


# ---------------------------------------------------------------------------
# Engine requirements (checked once, at startup)


@dataclass(frozen=True)
class EngineViolation:
    requirement: str
    detail: str
    stage: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"requirement": self.requirement, "detail": self.detail, "stage": self.stage}


class UnsupportedEngineError(RuntimeError):
    """The serving engine does not meet the LMSM execution contract."""

    def __init__(self, violations: Iterable[EngineViolation]):
        self.violations = list(violations)
        lines = "; ".join(
            f"{violation.requirement}: {violation.detail}" for violation in self.violations
        )
        super().__init__(
            "LMSM refuses to start: unsupported engine configuration "
            f"({len(self.violations)} violated requirement(s)): {lines}. "
            "--allow-unsupported-engine bypasses this check for tests only."
        )


_HOOK_NOTE = (
    "the Python activation hook does not run inside torch.compile graphs or "
    "CUDA-graph replay"
)

# How the runtime reads the declared activation: a Python forward hook (eager
# execution only) or the graph-captured ``lmsm::capture_rows`` operator.
OBSERVATION_HOOK = "hook"
OBSERVATION_GRAPH = "graph"
OBSERVATION_MODES = (OBSERVATION_HOOK, OBSERVATION_GRAPH)


def validate_observation_mode(value: str) -> str:
    if value not in OBSERVATION_MODES:
        raise ValueError(f"observation mode must be one of {OBSERVATION_MODES}, got {value!r}")
    return value


def check_engine_arguments(
    engine_args: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    vllm_version: str | None = None,
    observation_mode: str = OBSERVATION_HOOK,
    allow_async_scheduling: bool = False,
    allow_engine_process: bool = False,
    allow_partial_prefill: bool = False,
) -> list[EngineViolation]:
    """Check the arguments a runner passes to ``vllm.LLM`` before construction.

    Missing keys are treated as vLLM's defaults, several of which (prefix
    caching, chunked prefill, compiled execution) violate the contract.  With
    ``observation_mode="graph"`` the observation is a graph-captured operator,
    so compiled execution and CUDA graphs are allowed.  The installation of
    that operator is checked after construction instead.
    """

    environ = os.environ if environ is None else environ
    stage = "engine_arguments"
    validate_observation_mode(observation_mode)
    violations = _check_environment(environ, vllm_version, stage, allow_engine_process)
    if observation_mode == OBSERVATION_HOOK and not bool(
        engine_args.get("enforce_eager", False)
    ):
        violations.append(
            EngineViolation(
                "eager_execution",
                f"enforce_eager={engine_args.get('enforce_eager', '<unset>')}, so vLLM "
                "compiles the model with torch.compile and replays CUDA graphs; "
                f"{_HOOK_NOTE}",
                stage,
            )
        )
    if not allow_partial_prefill and engine_args.get("enable_prefix_caching", True):
        violations.append(
            EngineViolation(
                "prefix_caching_off",
                "enable_prefix_caching is not False (vLLM default: on); a cached "
                "prefix is not recomputed, so the prompt pass skips positions",
                stage,
            )
        )
    if not allow_partial_prefill and engine_args.get("enable_chunked_prefill", True):
        violations.append(
            EngineViolation(
                "chunked_prefill_off",
                "enable_chunked_prefill is not False (vLLM default: on); a chunked "
                "prompt is processed across several forward passes",
                stage,
            )
        )
    if not allow_async_scheduling and engine_args.get("async_scheduling", None) is not False:
        violations.append(
            EngineViolation(
                "async_scheduling_off",
                "async_scheduling is not False; asynchronous scheduling changes "
                "when sampled tokens become visible to the runtime",
                stage,
            )
        )
    if engine_args.get("speculative_config") is not None:
        violations.append(
            EngineViolation(
                "no_speculative_decoding",
                "speculative_config is set; one forward pass would process "
                "several unverified positions per request",
                stage,
            )
        )
    for key, requirement in (
        ("tensor_parallel_size", "tensor_parallel_size_1"),
        ("pipeline_parallel_size", "pipeline_parallel_size_1"),
        ("data_parallel_size", "data_parallel_size_1"),
    ):
        value = int(engine_args.get(key, 1) or 1)
        if value != 1:
            violations.append(
                EngineViolation(requirement, f"{key}={value}; LMSM requires 1", stage)
            )
    return violations


def check_vllm_config(
    vllm_config: Any,
    environ: Mapping[str, str] | None = None,
    vllm_version: str | None = None,
    observation_mode: str = OBSERVATION_HOOK,
    allow_async_scheduling: bool = False,
    allow_engine_process: bool = False,
    allow_partial_prefill: bool = False,
) -> list[EngineViolation]:
    """Check the engine's resolved ``VllmConfig`` (inside the worker).

    Only the device-evaluation path sets ``allow_async_scheduling``.  Its
    decisions and enforcement stay on the GPU, and its step indices come from
    its own per-request forward counters.  They do not depend on when sampled
    tokens reach the host.  Only that path sets ``allow_partial_prefill``
    (prefix caching, chunked prefill) as well.  It observes the prompt in the
    forward that completes the prompt, which vLLM always computes, so a cached
    or chunked prompt still delivers the declared observation.
    """

    environ = os.environ if environ is None else environ
    stage = "resolved_engine_config"
    validate_observation_mode(observation_mode)
    violations = _check_environment(environ, vllm_version, stage, allow_engine_process)

    model_config = getattr(vllm_config, "model_config", None)
    compilation = getattr(vllm_config, "compilation_config", None)
    enforce_eager = bool(getattr(model_config, "enforce_eager", False))
    mode = getattr(compilation, "mode", None)
    cudagraph_mode = getattr(compilation, "cudagraph_mode", None)
    mode_name = _enum_name(mode)
    cudagraph_name = _enum_name(cudagraph_mode)
    compiled = mode is not None and mode_name != "NONE" and _enum_value(mode) != 0
    graphs = cudagraph_mode is not None and cudagraph_name != "NONE"
    if observation_mode == OBSERVATION_HOOK and (not enforce_eager or compiled or graphs):
        violations.append(
            EngineViolation(
                "eager_execution",
                f"enforce_eager={enforce_eager}, compilation mode={mode_name}, "
                f"cudagraph mode={cudagraph_name}; {_HOOK_NOTE}",
                stage,
            )
        )

    cache_config = getattr(vllm_config, "cache_config", None)
    if not allow_partial_prefill and bool(getattr(cache_config, "enable_prefix_caching", False)):
        violations.append(
            EngineViolation(
                "prefix_caching_off",
                "enable_prefix_caching=True; a cached prefix is not recomputed, so "
                "the prompt pass skips positions",
                stage,
            )
        )

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if not allow_partial_prefill and bool(getattr(scheduler_config, "enable_chunked_prefill", False)):
        violations.append(
            EngineViolation(
                "chunked_prefill_off",
                "enable_chunked_prefill=True; a chunked prompt is processed across "
                "several forward passes",
                stage,
            )
        )
    if not allow_async_scheduling and bool(getattr(scheduler_config, "async_scheduling", False)):
        violations.append(
            EngineViolation(
                "async_scheduling_off",
                "async_scheduling=True; asynchronous scheduling changes when sampled "
                "tokens become visible to the runtime",
                stage,
            )
        )

    if getattr(vllm_config, "speculative_config", None) is not None:
        violations.append(
            EngineViolation(
                "no_speculative_decoding",
                "speculative decoding is configured; one forward pass would process "
                "several unverified positions per request",
                stage,
            )
        )

    parallel = getattr(vllm_config, "parallel_config", None)
    for key, requirement in (
        ("tensor_parallel_size", "tensor_parallel_size_1"),
        ("pipeline_parallel_size", "pipeline_parallel_size_1"),
        ("data_parallel_size", "data_parallel_size_1"),
    ):
        value = int(getattr(parallel, key, 1) or 1)
        if value != 1:
            violations.append(
                EngineViolation(requirement, f"{key}={value}; LMSM requires 1", stage)
            )
    return violations


def _check_environment(
    environ: Mapping[str, str],
    vllm_version: str | None,
    stage: str,
    allow_engine_process: bool = False,
) -> list[EngineViolation]:
    violations = []
    if vllm_version is None:
        vllm_version = _installed_vllm_version()
    if vllm_version is None or not str(vllm_version).startswith(SUPPORTED_VLLM_SERIES):
        violations.append(
            EngineViolation(
                "vllm_version_0_19",
                f"vLLM {vllm_version} is installed; the integration targets the "
                f"{SUPPORTED_VLLM_SERIES}x batch interface",
                stage,
            )
        )
    if _truthy(environ.get("VLLM_USE_V2_MODEL_RUNNER", "0")):
        violations.append(
            EngineViolation(
                "model_runner_v1",
                "VLLM_USE_V2_MODEL_RUNNER is set; Model Runner V2 changed the batch "
                "model the integration joins against",
                stage,
            )
        )
    if not allow_engine_process and environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        violations.append(
            EngineViolation(
                "in_process_engine_core",
                "VLLM_ENABLE_V1_MULTIPROCESSING="
                f"{environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset: 1>')}; the "
                "hook and the decision processor must share worker state with the "
                "release step",
                stage,
            )
        )
    return violations


def _installed_vllm_version() -> str | None:
    try:
        import vllm  # noqa: PLC0415

        return str(getattr(vllm, "__version__", None))
    except Exception:  # pragma: no cover - depends on the environment
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _enum_name(value: Any) -> str:
    if value is None:
        return "None"
    return str(getattr(value, "name", value))


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def parse_drop_observation_specs(specs: Iterable[str]) -> dict[str, frozenset[int]]:
    """Parse ``REQUEST_ID:STEP[,STEP...]`` test-injection specifications."""

    parsed: dict[str, set[int]] = {}
    for spec in specs:
        request_id, separator, steps = str(spec).rpartition(":")
        if not separator or not request_id or not steps:
            raise ValueError(
                f"--inject-drop-observation expects REQUEST_ID:STEP[,STEP...], got {spec!r}"
            )
        values = {int(step) for step in steps.split(",") if step.strip()}
        if not values or any(step < 0 for step in values):
            raise ValueError(f"invalid steps in {spec!r}")
        parsed.setdefault(request_id, set()).update(values)
    return {request_id: frozenset(steps) for request_id, steps in parsed.items()}


__all__ = [
    "OBSERVATION_GRAPH",
    "OBSERVATION_HOOK",
    "OBSERVATION_MODES",
    "validate_observation_mode",
    "ALLOW_WITH_RECORD",
    "COMPLETED",
    "DENY",
    "EXECUTION_FAILED",
    "EXECUTION_STATES",
    "EngineViolation",
    "ExecutionOutcome",
    "ExecutionRequirements",
    "NOT_REACHED",
    "NOT_REACHED_DISPOSITIONS",
    "ObservationWindow",
    "RELEASE_NONE",
    "RELEASE_ORDINARY",
    "RELEASE_REFUSAL",
    "RequestAccount",
    "UnsupportedEngineError",
    "assign_execution_state",
    "build_execution_record",
    "check_engine_arguments",
    "check_vllm_config",
    "derive_execution_requirements",
    "parse_drop_observation_specs",
    "release_outcome",
    "validate_not_reached_disposition",
]
