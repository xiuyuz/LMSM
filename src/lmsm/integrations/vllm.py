"""In-process vLLM 0.19 integration for profile-selected backends.

The integration supports offline serving with one worker, one activation
hook, no speculative decoding, and buffered output.  vLLM's ``BatchUpdate`` is
the only source of request identity.  The model-runner input metadata supplies
the current packed-token layout for each forward pass.  The hook site and
hidden width come from the active backend binding and the loaded model
configuration.

The execution contract (``lmsm.execution_contract``) adds a startup check of
the engine configuration, a per-request observation and evaluation account
keyed by the trusted request ID, exactly one execution state per finished
request, and a release step that uses that state.  A request whose evidence is
complete is released as its policy decided.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import wraps
from typing import Any, Callable

import numpy as np
import torch

from lmsm.execution_contract import (
    _StepSet,
    ALLOW_WITH_RECORD,
    COMPLETED,
    NOT_REACHED,
    OBSERVATION_GRAPH,
    OBSERVATION_HOOK,
    EngineViolation,
    RequestAccount,
    UnsupportedEngineError,
    assign_execution_state,
    build_execution_record,
    check_engine_arguments,  # noqa: F401 - re-exported for runners
    check_vllm_config,
    derive_execution_requirements,
    validate_observation_mode,
)
from lmsm.state import Decision


REQUEST_ID_EXTRA_ARG = "lmsm_request_id"
V1_MODEL_RUNNER = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
# Association ablation only: the policy-state key used for a packed batch row,
# and how a reused slot is treated when a new request is admitted to it.
ROW_STATE_KEY_PREFIX = "row-keyed-state:"
ROW_KEYED_MODES = ("reset_on_admission", "no_reset_on_admission")

try:
    _is_compiling = torch.compiler.is_compiling
except AttributeError:  # pragma: no cover - older torch
    def _is_compiling() -> bool:
        return False


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
    slot: int = -1  # device evaluation: the request's state slot


EVALUATION_HOST = "host"
EVALUATION_DEVICE = "device"
EVALUATION_MODES = (EVALUATION_HOST, EVALUATION_DEVICE)


@dataclass
class _PendingDeviceStep:
    """One forward's eligible rows, awaiting device results (device mode)."""

    epoch: int
    rows: np.ndarray
    slots: np.ndarray
    steps: np.ndarray
    flags: Any
    applied: bool = False
    has_result: bool = False
    event: Any = None
    dropped: list[tuple[int, int]] = field(default_factory=list)
    tokens: Any = None
    produced: bool = False
    gpu_start: Any = None


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
        # The vLLM module the activation key resolved to (Hugging Face names are
        # translated with the model's checkpoint-name mapper) and the point on
        # it: module input, or the residual stream after a decoder layer.
        self.resolved_module: str | None = None
        self.observation_point: str | None = None
        # Execution contract.
        self.requirements: Any | None = None
        self.accounts: dict[str, RequestAccount] = {}
        self.execution_records: dict[str, dict[str, Any]] = {}
        self.engine_violations: list[EngineViolation] = []
        self.engine_check: dict[str, Any] | None = None
        self.gate_constructions = 0
        self.model_runner_class: str | None = None
        self.idle_hook_calls = 0
        self.idle_observed_width: int | None = None
        self.evaluation_errors = 0
        # Requests failed closed because evidence or a score was NaN or infinite.
        self.nonfinite_failures = 0
        self.uncaptured_forwards = 0
        # Knobs for the mediation tests, off by default and recorded in run metadata.
        self.allow_unsupported_engine = False
        self.inject_drop_observations: dict[str, frozenset[int]] = {}
        self.inject_row_keyed_state = False
        self.row_keyed_mode: str | None = None
        self.initialized_slots: set[str] = set()
        self.state_provenance: dict[str, set[str]] = {}
        # Graph-compatible observation (``observation_mode="graph"``).
        self.observation_mode = OBSERVATION_HOOK
        self.graph_capture_disabled_for_test = False
        self.num_forward_tokens = 0
        self.graph_epoch_checks = 0
        self.graph_epoch_failures = 0
        # Device-resident evaluation (``evaluation_mode="device"``).
        self.evaluation_mode = EVALUATION_HOST
        self.device_engine: Any | None = None
        self.runner: Any | None = None
        self.logits_indices: torch.Tensor | None = None
        self.pending_step: _PendingDeviceStep | None = None
        self.device_archive: dict[str, dict[str, Any]] = {}
        self.debug_checks = bool(os.environ.get("LMSM_DEBUG_CHECKS"))
        self.profile_steps = bool(os.environ.get("LMSM_PROFILE_STEPS"))
        self.profile_totals: dict[str, float] = {}
        self.profile_events: list[tuple[Any, Any]] = []
        self.planned_counts: np.ndarray | None = None
        self.planned_steps: np.ndarray | None = None
        self.planned_partial: np.ndarray | None = None
        self.planned_prompt_complete: np.ndarray | None = None
        self.planned_epoch = -1
        # Test only: record every delivered observation row (host paths).
        self.dump_observations = bool(os.environ.get("LMSM_DUMP_OBSERVATIONS"))
        self.dumped_observations: list[tuple[list[str], list[int], Any]] = []
        # Test only: called with (request_ids, steps, rows, ok, engine) after
        # every device evaluation (request-association tests).
        self.device_observer: Callable[..., None] | None = None
        self.device_steps = 0
        self.device_flushes = 0
        self.async_scheduling = False
        self.discarded_async_decisions = 0
        # Requests whose only non-finite evidence was on a discarded forward.
        self.discarded_async_nonfinite = 0
        self.remote_worker_pid: int | None = None

    def configure(
        self,
        evaluator: Any,
        fixed_refusal_tokens: list[int],
        *,
        allow_unsupported_engine: bool = False,
        inject_drop_observations: dict[str, Any] | None = None,
        inject_row_keyed_state: bool | str = False,
        observation_mode: str = OBSERVATION_HOOK,
        graph_capture_disabled_for_test: bool = False,
        evaluation_mode: str = EVALUATION_HOST,
    ) -> None:
        """Configure the worker state.

        The keyword arguments are knobs for the mediation tests and are off by
        default.  ``allow_unsupported_engine`` records startup-check violations
        instead of refusing to start.  ``inject_drop_observations`` withholds
        the observation of the listed request IDs at the listed steps.
        ``inject_row_keyed_state`` keys policy state by packed batch row
        instead of the trusted request ID, for the request-association
        ablation.  Its mode ``reset_on_admission``, also selected by ``True``,
        gives each admitted request a fresh slot but does not move slot state
        with a moved request.  ``no_reset_on_admission`` also lets an admitted
        request inherit the slot's previous state.
        """

        self.reset()
        self.evaluator = evaluator
        self.fixed_refusal_tokens = list(fixed_refusal_tokens)
        binding_ids = sorted({rule.binding_id for rule in evaluator.profile.active_rules})
        bindings = [evaluator.profile.binding_for(binding_id) for binding_id in binding_ids]
        keys = {binding.activation_key for binding in bindings}
        widths = {binding.config.get("hidden_width") for binding in bindings}
        if len(keys) != 1 or len(widths) != 1:
            # Several bindings (a composed policy) must read one activation: one capture.
            raise ValueError("LMSM vLLM requires the active rules to read one activation key")
        binding = bindings[0]
        self.activation_key = binding.activation_key
        configured_width = binding.config.get("hidden_width")
        if configured_width is not None:
            self.expected_hidden_width = int(configured_width)
        self.requirements = derive_execution_requirements(evaluator)
        self.allow_unsupported_engine = bool(allow_unsupported_engine)
        self.inject_drop_observations = {
            str(request_id): frozenset(int(step) for step in steps)
            for request_id, steps in (inject_drop_observations or {}).items()
        }
        if inject_row_keyed_state is True:
            inject_row_keyed_state = ROW_KEYED_MODES[0]
        if inject_row_keyed_state and inject_row_keyed_state not in ROW_KEYED_MODES:
            raise ValueError(f"row-keyed ablation mode must be one of {ROW_KEYED_MODES}")
        self.row_keyed_mode = inject_row_keyed_state or None
        self.inject_row_keyed_state = self.row_keyed_mode is not None
        self.observation_mode = validate_observation_mode(observation_mode)
        self.graph_capture_disabled_for_test = bool(graph_capture_disabled_for_test)
        if self.graph_capture_disabled_for_test and self.observation_mode != OBSERVATION_GRAPH:
            raise ValueError("graph_capture_disabled_for_test requires observation_mode='graph'")
        if evaluation_mode not in EVALUATION_MODES:
            raise ValueError(f"evaluation mode must be one of {EVALUATION_MODES}")
        self.evaluation_mode = evaluation_mode
        if evaluation_mode == EVALUATION_DEVICE:
            if self.observation_mode != OBSERVATION_GRAPH:
                raise ValueError("device evaluation requires observation_mode='graph'")
            if self.inject_row_keyed_state:
                raise ValueError("the row-keyed association ablation uses host evaluation")
            from lmsm.integrations.vllm_device import check_device_evaluator

            check_device_evaluator(evaluator)

    # ------------------------------------------------------------------
    # Device-resident evaluation

    def init_device(self, max_rows: int, max_steps: int, device: torch.device) -> None:
        """Allocate the device engine and host slot tables (once per process)."""

        if self.device_engine is not None:
            return
        from lmsm.integrations.vllm_device import build_device_engine

        num_slots = int(max_rows)
        self.device_engine = build_device_engine(
            self.evaluator, num_slots, int(max_rows), max_steps, device
        )
        self.device = device
        self.max_rows = int(max_rows)
        self.slot_of_row = np.full(self.max_rows, -1, dtype=np.int64)
        self.slot_request: list[str | None] = [None] * num_slots
        self.slot_state: list[RequestState | None] = [None] * num_slots
        self.free_slots = list(range(num_slots - 1, -1, -1))
        self.slot_by_request: dict[str, int] = {}
        self.device_event = torch.cuda.Event() if torch.cuda.is_available() else None
        self.slot_prompt_len = np.zeros(num_slots, dtype=np.int64)
        self.slot_actioned = np.zeros(num_slots, dtype=bool)
        self.slot_failed = np.zeros(num_slots, dtype=bool)
        self.slot_needs_prompt = np.zeros(num_slots, dtype=bool)
        self.slot_decision_step = np.full(num_slots, -1, dtype=np.int64)
        # Forwards that included each request so far: the step index of its
        # next forward (0 = prompt pass).  Independent of when sampled tokens
        # reach the host, so it also holds under async scheduling.
        self.slot_step = np.zeros(num_slots, dtype=np.int64)
        pin = torch.cuda.is_available()
        self.reset_stage = torch.zeros((self.max_rows,), dtype=torch.int64, pin_memory=pin)
        self.reset_index = torch.zeros((self.max_rows,), dtype=torch.int64, device=device)
        self.reset_event = torch.cuda.Event() if torch.cuda.is_available() else None
        windows = self.requirements.windows if self.requirements is not None else ()
        self.window_bounds = [
            (window.first_step, window.last_step) for window in windows
        ]

    def _device_fail(self, slot: int, step: int | None, reason: str) -> None:
        request_id = self.slot_request[slot]
        if request_id is None:
            return
        account = self.accounts.get(request_id)
        if account is not None:
            account.fail(step, reason)
        self.slot_failed[slot] = True

    def _update_batch_device(self, batch_update: Any | None) -> None:
        started = self._profile_start()
        try:
            self._update_batch_device_inner(batch_update)
        finally:
            self._profile_stop("cpu_update_batch", started)

    def _update_batch_device_inner(self, batch_update: Any | None) -> None:
        if batch_update is None:
            # Nothing leaves or joins the batch: the previous step's results
            # are flushed after the next forward launch (off the critical path).
            return
        self._flush_device()
        released: list[tuple[str, int, int]] = []
        new_slots: list[int] = []

        def release(state: RequestState) -> None:
            if state.slot >= 0 and self.slot_state[state.slot] is state:
                released.append((state.request_id, state.slot, len(state.output_token_ids)))

        for row in batch_update.removed:
            removed = self.rows.pop(int(row), None)
            self.slot_of_row[int(row)] = -1
            if removed is not None:
                release(removed)
        self._archive_released(released)

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
                release(previous)
                self._archive_released(released)
            state = RequestState(
                request_id=str(request_id),
                prompt_token_ids=list(prompt_ids or []),
                output_token_ids=output_ids,
            )
            self.rows[row] = state
            for store in (
                self.archived_feature_maxima,
                self.archived_policy_scores,
                self.archived_policy_score_steps,
                self.archived_policy_actions,
                self.device_archive,
            ):
                store.pop(state.request_id, None)
            slot = self._allocate_slot(state)
            new_slots.append(slot)
            self.slot_of_row[row] = slot
            self._admit(state, row)
            account = self.accounts.get(state.request_id)
            self.slot_failed[slot] = bool(account is not None and account.failed)

        for source, destination, directionality in batch_update.moved:
            source, destination = int(source), int(destination)
            if directionality.name == "SWAP":
                source_state = self.rows.pop(source, None)
                destination_state = self.rows.pop(destination, None)
                if source_state is not None:
                    self.rows[destination] = source_state
                if destination_state is not None:
                    self.rows[source] = destination_state
                self.slot_of_row[[source, destination]] = self.slot_of_row[
                    [destination, source]
                ]
            else:
                state = self.rows.pop(source, None)
                if state is not None:
                    overwritten = self.rows.get(destination)
                    if overwritten is not None and overwritten is not state:
                        # condense() fills a departed request's row with this
                        # move, and the departure is visible only here.
                        release(overwritten)
                    self.rows[destination] = state
                    self.slot_of_row[destination] = self.slot_of_row[source]
                self.slot_of_row[source] = -1

        for row in list(self.rows):
            if row >= int(batch_update.batch_size):
                removed = self.rows.pop(row)
                self.slot_of_row[row] = -1
                release(removed)
        self._archive_released(released)
        if new_slots:
            count = len(new_slots)
            if self.reset_event is not None:
                # The previous reset copy must have left the pinned buffer.
                self.reset_event.synchronize()
            self.reset_stage[:count] = torch.tensor(new_slots, dtype=torch.int64)
            index = self.reset_index[:count]
            index.copy_(self.reset_stage[:count], non_blocking=True)
            self.device_engine.reset_slots(index)
            if self.reset_event is not None:
                self.reset_event.record()

    def _slot_for(self, request_id: str) -> int | None:
        slot = self.slot_by_request.get(request_id)
        if slot is not None and self.slot_request[slot] == request_id:
            return slot
        return None

    def _allocate_slot(self, state: RequestState) -> int:
        if not self.free_slots:
            raise AssertionError("no free LMSM device slot (more live requests than rows)")
        slot = self.free_slots.pop()
        state.slot = slot
        self.slot_request[slot] = state.request_id
        self.slot_state[slot] = state
        self.slot_by_request[state.request_id] = slot
        self.slot_prompt_len[slot] = len(state.prompt_token_ids)
        self.slot_actioned[slot] = False
        self.slot_failed[slot] = False
        self.slot_needs_prompt[slot] = True
        self.slot_decision_step[slot] = -1
        self.slot_step[slot] = len(state.output_token_ids)
        return slot

    def _archive_released(self, released: list[tuple[str, int, int]]) -> None:
        """Copy departing requests' device state to the host and free their slots."""

        if not released:
            return
        pending = [
            (request_id, slot, length)
            for request_id, slot, length in released
            if request_id not in self.device_archive
        ]
        if pending:
            records = self.device_engine.archive(
                [slot for _, slot, _ in pending], [length for _, _, length in pending]
            )
            for (request_id, _, _), record in zip(pending, records, strict=True):
                self._store_archive(request_id, record)
        for request_id, slot, _ in released:
            if self.slot_by_request.get(request_id) == slot:
                del self.slot_by_request[request_id]
            self.slot_request[slot] = None
            self.slot_state[slot] = None
            self.free_slots.append(slot)
        released.clear()

    def _store_archive(self, request_id: str, record: dict[str, Any]) -> None:
        self.device_archive[request_id] = record
        self.archived_feature_maxima[request_id] = record.get("feature_maxima", {})
        self.archived_policy_scores[request_id] = record["policy_scores"]
        self.archived_policy_score_steps[request_id] = record["policy_score_steps"]
        self.archived_policy_actions[request_id] = record["policy_actions"]

    def _ensure_device_archived(self, request_id: str) -> None:
        if self.evaluation_mode != EVALUATION_DEVICE:
            return
        self._flush_device()
        if request_id in self.device_archive:
            return
        slot = self._slot_for(request_id)
        if slot is None:
            return
        state = self.slot_state[slot]
        length = len(state.output_token_ids) if state is not None else 0
        record = self.device_engine.archive([slot], [length])[0]
        self._store_archive(request_id, record)

    def _on_discarded_forward(self, step: int, final_length: int) -> bool:
        """Whether a device-mode step belongs to a forward vLLM discarded.

        A request that generated ``N`` tokens consumed steps ``0 .. N - 1``.
        Under async scheduling a request that already finished can be
        scheduled for one more forward, whose sampled token vLLM discards.
        That forward processes step ``N`` or later.  Decisions and non-finite
        evidence use this one boundary.  Without async scheduling no forward is
        discarded, so a step at or past the final length is an anomaly that the
        account fails closed.
        """

        return self.async_scheduling and step >= final_length

    def finalize_device_request(self, request_id: str, ordinary_output_tokens: list[int]) -> None:
        """Release-time completion of a device-mode request's action and account.

        Fills the action record's prefix from the final output.  Under async
        scheduling a request that already finished can be scheduled for one
        more forward whose sampled token vLLM discards.  A decision recorded at
        or after the final length belongs to that discarded forward.  It is
        removed, since it never affected the released output.
        """

        if self.evaluation_mode != EVALUATION_DEVICE:
            return
        self._flush_device()
        count = len(ordinary_output_tokens)
        account = self.accounts.get(request_id)
        for index in range(len(self.action_records) - 1, -1, -1):
            record = self.action_records[index]
            if record.request_id != request_id:
                continue
            if self._on_discarded_forward(record.step, count):
                del self.action_records[index]
                self.discarded_async_decisions += 1
                if account is not None and account.decision_step == record.step:
                    account.decision_step = None
                    account.decision = None
                continue
            prefix = list(ordinary_output_tokens[: record.step])
            record.ordinary_model_prefix_tokens = prefix
            record.admitted_prefix_tokens = list(prefix)
            break

    def _materialize_device_account(
        self, request_id: str, generated_token_count: int | None = None
    ) -> None:
        """Fill the execution account from the device's own records.

        A step counts as observed only if the device engine's epoch check passed
        for it.  The anytime and selected-feature engines evaluate in the same
        step function that writes that observation, so an observed step is also
        an evaluated one.  The checkpoint engine evaluates its fusion in a
        separate phase and archives which evaluation points that phase completed
        (``evaluated_points``).  A declared evaluation point it did not complete
        stays unevaluated, so the request fails closed.  An engine that met a
        NaN or infinite observation, pooled mean or score archives the first
        such step (``nonfinite_step``), and the request fails closed there too.
        The exception is a step on a forward that vLLM discarded, the boundary
        that ``finalize_device_request`` applies to decisions.  That step is
        the first non-finite one, so every consumed step was finite.  The
        account keeps it as ``discarded_nonfinite_step``, a diagnostic and not
        a failure.  Without ``generated_token_count`` no step counts as
        discarded.
        """

        self._ensure_device_archived(request_id)
        record = self.device_archive.get(request_id)
        account = self.accounts.get(request_id)
        if record is None or account is None or record.get("materialized"):
            return
        observed = _StepSet()
        observed._bits = bytearray(np.asarray(record["observed"], dtype=np.uint8).tobytes())
        evaluated = _StepSet()
        evaluated._bits = bytearray(observed._bits)
        points = record.get("evaluated_points")
        steps = self.requirements.evaluation_steps if self.requirements is not None else None
        if points is not None and steps:
            completed = {int(step) for step in points}
            for step in steps:
                if step not in completed:
                    evaluated.discard(step)
        account.observed = observed
        account.evaluated = evaluated
        if record.get("evaluation_error"):
            self.evaluation_errors += 1
            account.fail(None, f"policy evaluation failed: {record['evaluation_error']}")
        nonfinite_step = record.get("nonfinite_step")
        if nonfinite_step is not None:
            nonfinite_step = int(nonfinite_step)
            if generated_token_count is not None and self._on_discarded_forward(
                nonfinite_step, int(generated_token_count)
            ):
                self.discarded_async_nonfinite += 1
                account.discarded_nonfinite_step = nonfinite_step
            else:
                self.nonfinite_failures += 1
                account.fail(nonfinite_step, _nonfinite_reason(nonfinite_step))
        record["materialized"] = True

    def _requires_observation_vector(
        self, steps: np.ndarray, decision_steps: np.ndarray
    ) -> np.ndarray:
        required = np.zeros(steps.shape, dtype=bool)
        for first, last in self.window_bounds:
            inside = steps >= first
            if last is not None:
                inside &= steps <= last
            required |= inside
        required &= ~((decision_steps >= 0) & (steps > decision_steps))
        return required

    def _begin_forward_device(self, num_scheduled_tokens: Any) -> None:
        """Before the forward: stamp the expected capture epoch, nothing else.

        Everything that only the policy needs (eligibility, accounting, index
        staging) runs in ``apply_device``, after the forward has been launched,
        so it overlaps the GPU instead of delaying the launch.
        """

        started = self._profile_start()
        counts = np.array(num_scheduled_tokens, dtype=np.int64)
        n = int(counts.shape[0])
        slots = self.slot_of_row[:n]
        if n and int(slots.min()) < 0:
            raise AssertionError("current post-update batch has an unmapped request row")
        # A forward that schedules only part of a prompt (chunked prefill)
        # processes no declared position.  The forward that completes the
        # prompt is step 0.  With prefix caching, vLLM always recomputes at
        # least the last prompt token.  Every later forward processes one token.
        computed = self.runner.input_batch.num_computed_tokens_cpu[:n].astype(np.int64)
        prompt_len = self.slot_prompt_len[slots]
        in_prompt = computed < prompt_len
        completes = in_prompt & (computed + counts >= prompt_len)
        self.planned_partial = in_prompt & ~completes
        self.planned_prompt_complete = completes
        self.planned_steps = self.slot_step[slots].copy()
        self.slot_step[slots] += (~self.planned_partial).astype(np.int64)
        self.forward_epoch += 1
        self.num_forward_tokens = int(counts.sum())
        from lmsm.integrations.vllm_graph import CAPTURE

        CAPTURE.set_step_epoch(self.forward_epoch)
        self.planned_counts = counts
        self.planned_epoch = self.forward_epoch
        self._profile_stop("cpu_begin_forward", started)

    def _plan_device_step(self) -> "_PendingDeviceStep | None":
        counts = self.planned_counts
        if counts is None or self.planned_epoch != self.forward_epoch:
            return None
        self.planned_counts = None
        n = int(counts.shape[0])
        if n and int(counts.min()) <= 0:
            raise AssertionError("each active vLLM request must schedule at least one token")
        slots = self.slot_of_row[:n]
        steps = self.planned_steps
        if self.debug_checks and not self.async_scheduling:
            batch = self.runner.input_batch
            from_batch = batch.num_tokens_no_spec[:n].astype(np.int64) - batch.num_prompt_tokens[:n]
            if not np.array_equal(from_batch, steps):
                raise AssertionError("per-request forward counters disagree with the input batch")
        if self.debug_checks and not self.async_scheduling:
            expected = np.asarray(
                [len(self.rows[row].output_token_ids) for row in range(n)], dtype=np.int64
            )
            if not np.array_equal(expected, steps):
                raise AssertionError("vectorized step counts disagree with the request lists")
            for row in range(n):
                if self.slot_request[slots[row]] != self.rows[row].request_id:
                    raise AssertionError("row-to-slot map disagrees with the request rows")
        engine = self.device_engine
        partial = self.planned_partial
        prompt_endpoint = (
            engine.observes_prompt
            & (steps == 0)
            & self.planned_prompt_complete
            & self.slot_needs_prompt[slots]
        )
        decode_endpoint = engine.observes_decode & (steps > 0) & ~self.slot_actioned[slots] & ~partial
        if engine.max_decode_step is not None:
            decode_endpoint &= steps <= engine.max_decode_step
        eligible = prompt_endpoint | decode_endpoint
        failed = self.slot_failed[slots]
        required = self._requires_observation_vector(steps, self.slot_decision_step[slots])
        required &= ~partial
        for row in np.nonzero(required & ~eligible & ~failed)[0]:
            row = int(row)
            step = int(steps[row])
            if step == 0 and counts[row] != self.slot_prompt_len[slots[row]]:
                reason = (
                    f"the prompt pass scheduled {int(counts[row])} of "
                    f"{int(self.slot_prompt_len[slots[row]])} prompt tokens (chunked "
                    "prefill or prefix caching), so the required prompt observation "
                    "was not captured"
                )
            else:
                reason = f"the required observation at step {step} was not scheduled"
            self._device_fail(int(slots[row]), step, reason)
        eligible &= ~self.slot_failed[slots]
        rows = np.nonzero(eligible)[0]
        self.skip_capture = rows.size == 0
        if rows.size == 0:
            return None
        row_slots = slots[rows]
        row_steps = steps[rows]
        self.slot_needs_prompt[row_slots[row_steps == 0]] = False
        dropped: list[tuple[int, int]] = []
        if self.inject_drop_observations:
            keep = np.ones(rows.shape, dtype=bool)
            for position, (slot, step) in enumerate(zip(row_slots, row_steps, strict=True)):
                request_id = self.slot_request[int(slot)]
                if int(step) in self.inject_drop_observations.get(request_id, ()):
                    keep[position] = False
                    dropped.append((int(slot), int(step)))
                    account = self.accounts.get(request_id)
                    if account is not None:
                        account.injected_faults.append(
                            {"kind": "drop_observation", "step": int(step), "path": "device"}
                        )
            rows, row_slots, row_steps = rows[keep], row_slots[keep], row_steps[keep]
        from lmsm.integrations.vllm_device import StepFlags

        flags = StepFlags(
            any_prompt=bool((row_steps == 0).any()),
            any_window_1_end=bool(
                getattr(engine, "split", None) is not None
                and (row_steps == engine.split).any()
            ),
            any_evaluation=bool(
                getattr(engine, "evaluation_step", None) is None
                or (row_steps == engine.evaluation_step).any()
            ),
        )
        pending = _PendingDeviceStep(
            epoch=self.forward_epoch,
            rows=rows,
            slots=row_slots,
            steps=row_steps,
            flags=flags,
            dropped=dropped,
        )
        # Packed position of each eligible row's final scheduled token (the
        # same index vLLM samples from: query_start_loc[1:] - 1).
        pending.tokens = (np.cumsum(counts) - 1)[rows]
        self.pending_step = pending
        return pending

    def _profile_start(self) -> float | None:
        if not self.profile_steps:
            return None
        import time

        return time.perf_counter()

    def _profile_stop(self, key: str, started: float | None) -> None:
        if started is None:
            return
        import time

        self.profile_totals[key] = self.profile_totals.get(key, 0.0) + (
            time.perf_counter() - started
        )
        self.profile_totals[key + "_calls"] = self.profile_totals.get(key + "_calls", 0) + 1

    def apply_device(self, logits: torch.Tensor) -> torch.Tensor:
        started = self._profile_start()
        if started is not None:
            idle = torch.cuda.current_stream().query()
            key = "apply_entered_with_idle_gpu"
            self.profile_totals[key] = self.profile_totals.get(key, 0) + int(idle)
        try:
            return self._apply_device(logits)
        finally:
            self._profile_stop("cpu_apply", started)

    def after_forward(self) -> None:
        """Right after the forward launch: flush the previous step, then
        evaluate this step on the device.

        The host work here overlaps the forward on the GPU, and the device step
        runs between the forward and the logits computation, so the policy
        adds no host synchronization and almost no time to the step.
        """

        started = self._profile_start()
        try:
            flush_started = self._profile_start()
            self._flush_device()
            self._profile_stop("cpu_flush", flush_started)
            self._launch_device_step()
        finally:
            self._profile_stop("cpu_after_forward", started)

    def _launch_device_step(self) -> None:
        plan_started = self._profile_start()
        pending = self._plan_device_step()
        self._profile_stop("cpu_plan", plan_started)
        if pending is None or pending.applied:
            return
        pending.applied = True
        m = int(pending.rows.shape[0])
        if m == 0:
            return
        engine = self.device_engine
        if engine.capture_buffer is None:
            from lmsm.integrations.vllm_graph import CAPTURE

            assert CAPTURE.buffer is not None and CAPTURE.captured_epoch is not None
            engine.bind_capture(CAPTURE.buffer, CAPTURE.captured_epoch)
            engine.instrument = bool(self.device_observer is not None or self.dump_observations)
            if engine.instrument and not engine.use_graphs and engine.last_hidden is None:
                engine.last_hidden = torch.zeros(
                    (engine.max_rows, CAPTURE.buffer.shape[1]),
                    dtype=CAPTURE.buffer.dtype,
                    device=engine.device,
                )
            engine.capture_graphs()
        if self.debug_checks and self.logits_indices is not None:
            expected = self.logits_indices.index_select(
                0, torch.from_numpy(pending.rows).to(self.logits_indices.device)
            )
            if not np.array_equal(expected.cpu().numpy(), pending.tokens):
                raise AssertionError("final-token positions disagree with vLLM's logits indices")
        if self.profile_steps:
            pending.gpu_start = torch.cuda.Event(enable_timing=True)
            pending.gpu_start.record()
        self.device_steps += 1
        load_started = self._profile_start()
        engine.load(pending.rows, pending.slots, pending.steps, pending.tokens, pending.epoch)
        self._profile_stop("cpu_load", load_started)
        run_started = self._profile_start()
        pending.produced = engine.run(pending.flags)
        self._profile_stop("cpu_run", run_started)
        if engine.instrument:
            hidden = engine.last_hidden[:m]
            request_ids = [self.slot_request[int(slot)] for slot in pending.slots]
            if self.device_observer is not None:
                self.device_observer(
                    request_ids, [int(step) for step in pending.steps], hidden, engine.ok, engine
                )
            if self.dump_observations:
                self.dumped_observations.append(
                    (
                        [str(request_id) for request_id in request_ids],
                        [int(step) for step in pending.steps],
                        hidden.detach().float().cpu().numpy(),
                    )
                )
        if self.profile_steps:
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_end.record()
            self.profile_events.append((pending.gpu_start, gpu_end))
            pending.event = gpu_end
        else:
            pending.event = self.device_event
            pending.event.record()

    def _apply_device(self, logits: torch.Tensor) -> torch.Tensor:
        if self.planned_counts is not None:
            # The forward hook did not run (unexpected): evaluate here instead.
            self._flush_device()
            self._launch_device_step()
        pending = self.pending_step
        if pending is None or pending.epoch != self.forward_epoch or not pending.produced:
            return logits
        eos = self.eos_token_id
        if eos is None:
            raise AssertionError("termination token is not configured")
        act = self.device_engine.act_rows[: logits.shape[0]]
        eos_column = logits[:, eos].clone()
        logits.masked_fill_(act.unsqueeze(1), float("-inf"))
        logits[:, eos] = torch.where(act, torch.zeros_like(eos_column), eos_column)
        pending.has_result = True
        pending.event.record()
        return logits

    def _flush_device(self) -> None:
        pending = self.pending_step
        if pending is None:
            return
        if not pending.applied:
            # The policy never ran for this forward: fail closed.
            for slot, step in zip(pending.slots, pending.steps, strict=True):
                self._device_fail(
                    int(slot), int(step), "the policy evaluation did not run for this forward"
                )
            self.pending_step = None
            return
        self.pending_step = None
        m = int(pending.rows.shape[0])
        if m == 0:
            return
        self.device_flushes += 1
        pending.event.synchronize()
        self.graph_epoch_checks += 1
        if not bool(self.device_engine.pinned_ok[0]):
            self.graph_epoch_failures += 1
            self.uncaptured_forwards += 1
            for slot, step in zip(pending.slots, pending.steps, strict=True):
                self._device_fail(
                    int(slot),
                    int(step),
                    "the activation capture did not run in the forward pass, so no "
                    "observation was delivered",
                )
        if not (pending.has_result or pending.produced):
            return
        result = self.device_engine.pinned_result[:m].numpy()
        engine = self.device_engine
        profile = self.evaluator.profile
        for position in np.nonzero(result[:, 0] > 0.5)[0]:
            slot = int(pending.slots[position])
            step = int(pending.steps[position])
            state = self.slot_state[slot]
            if state is None or state.actioned:
                continue
            rule_index = int(result[position, 1])
            target_id = engine.target_ids[rule_index]
            rule = profile.rule_for_target(target_id)
            action = profile.action_for_rule(rule.rule_id)
            # The prefix is the request's first `step` tokens.  It is filled from
            # the final output at release (``_finalize_device_actions``), which
            # under async scheduling may not be on the host yet.
            prefix: list[int] = []
            released = list(self.fixed_refusal_tokens) if action == "refuse" else []
            self.action_records.append(
                ActionRecord(
                    request_id=state.request_id,
                    step=step,
                    action=action,
                    rule_id=rule.rule_id,
                    target_id=target_id,
                    score=float(result[position, 2]),
                    threshold=engine.threshold_for(rule_index),
                    policy_name=profile.policy.name,
                    policy_version=profile.policy.version,
                    rule_library_version=profile.rule_library.version,
                    active_rule_ids=list(profile.policy.active_rule_ids),
                    ordinary_model_prefix_tokens=prefix,
                    forced_termination_token=int(self.eos_token_id),
                    fixed_refusal_tokens=list(self.fixed_refusal_tokens),
                    admitted_prefix_tokens=prefix,
                    released_output_tokens=released,
                )
            )
            state.actioned = True
            state.step = step
            self.slot_actioned[slot] = True
            self.slot_decision_step[slot] = step
            account = self.accounts.get(state.request_id)
            if account is not None:
                account.record_decision(step, action)

    def update_batch(self, batch_update: Any | None) -> None:
        if self.evaluation_mode == EVALUATION_DEVICE:
            self._update_batch_device(batch_update)
            return
        if batch_update is None:
            return

        for row in batch_update.removed:
            removed = self.rows.pop(int(row), None)
            if removed is not None:
                self._archive_and_remove(removed.request_id, int(row))

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
                self._archive_and_remove(previous.request_id, row)
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
            key = self._state_key(state.request_id, row)
            if self.row_keyed_mode == "no_reset_on_admission":
                # Ablation: a slot's state is created once and then inherited
                # by every request admitted to that row.
                if key not in self.initialized_slots:
                    self.evaluator.add(key)
                    self.initialized_slots.add(key)
            else:
                if self.inject_row_keyed_state:
                    # Ablation: reset the slot for each admitted request, but
                    # never move slot state with a moved request.  (``add``
                    # alone assumes a prior ``remove``.)
                    self.evaluator.remove(key)
                self.evaluator.add(key)
            self._admit(state, row)

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
                    overwritten = self.rows.get(destination)
                    if overwritten is not None and self.inject_row_keyed_state:
                        # vLLM's condense() consumes a finished request's slot
                        # from ``removed`` and fills it with this move, so the
                        # departure is visible only here.  Request-keyed state
                        # needs no action.  The row-keyed ablation archives the
                        # departed request's slot state before it is inherited.
                        self._archive_and_remove(overwritten.request_id, destination)
                    self.rows[destination] = state

        for row in list(self.rows):
            if row >= int(batch_update.batch_size):
                removed = self.rows.pop(row)
                self._archive_and_remove(removed.request_id, row)

    def _state_key(self, request_id: str, row: int | None) -> str:
        """Return the policy-state key: the trusted request ID.

        Only the association ablation (``inject_row_keyed_state``) keys state
        by the packed batch row instead.
        """

        if self.inject_row_keyed_state and row is not None:
            return f"{ROW_STATE_KEY_PREFIX}{row}"
        return request_id

    def _live_state_key(self, request_id: str) -> str:
        if self.inject_row_keyed_state:
            for row, request in self.rows.items():
                if request.request_id == request_id:
                    return self._state_key(request_id, row)
        return request_id

    def _admit(self, state: RequestState, row: int) -> None:
        """Open, or mark, the execution account for an admitted request."""

        request_id = state.request_id
        step = len(state.output_token_ids)
        account = self.accounts.get(request_id)
        if step == 0 or account is None:
            # A new request.  If it arrives with generated tokens, the steps
            # before admission were never observed, and the completion check
            # against the output length accounts for them.
            account = RequestAccount(request_id, len(state.prompt_token_ids))
            self.accounts[request_id] = account
            if any(
                other_row != row and other.request_id == request_id
                for other_row, other in self.rows.items()
            ):
                account.fail(step, "request ID is already in flight in another batch row")
        else:
            account.admissions += 1
            account.fail(
                step,
                f"re-admitted to the batch after {step} generated token(s) "
                "(preemption or descheduling); its policy state was reset, so "
                "earlier observations no longer inform the decision",
            )
        if self.inject_row_keyed_state:
            key = self._state_key(request_id, row)
            account.state_key = key
            if self.row_keyed_mode == "no_reset_on_admission":
                self.state_provenance.setdefault(key, set())
            else:
                self.state_provenance[key] = set()

    def _archive_and_remove(self, request_id: str, row: int | None = None) -> None:
        key = self._state_key(request_id, row)
        self.archived_feature_maxima[request_id] = self.evaluator.feature_maxima_for(
            key
        )
        scores_for = getattr(self.evaluator, "scores_for", None)
        score_steps_for = getattr(self.evaluator, "score_steps_for", None)
        actions_for = getattr(self.evaluator, "actions_for", None)
        self.archived_policy_scores[request_id] = (
            scores_for(key) if scores_for is not None else {}
        )
        self.archived_policy_score_steps[request_id] = (
            score_steps_for(key) if score_steps_for is not None else {}
        )
        self.archived_policy_actions[request_id] = (
            actions_for(key) if actions_for is not None else {}
        )
        if not self.inject_row_keyed_state:
            self.evaluator.remove(request_id)
        # Under the row-keyed ablation a slot keeps its state until the next
        # request is admitted to that row, and moved requests inherit it.

    def feature_maxima_for(self, request_id: str) -> dict[str, dict[int, float]]:
        self._ensure_device_archived(request_id)
        if request_id in self.archived_feature_maxima:
            return self.archived_feature_maxima[request_id]
        return self.evaluator.feature_maxima_for(self._live_state_key(request_id))

    def policy_scores_for(self, request_id: str) -> dict[str, Any]:
        self._ensure_device_archived(request_id)
        if request_id in self.archived_policy_scores:
            return dict(self.archived_policy_scores[request_id])
        scores_for = getattr(self.evaluator, "scores_for", None)
        return (
            scores_for(self._live_state_key(request_id))
            if scores_for is not None
            else {}
        )

    def policy_actions_for(self, request_id: str) -> dict[str, bool]:
        self._ensure_device_archived(request_id)
        if request_id in self.archived_policy_actions:
            return dict(self.archived_policy_actions[request_id])
        actions_for = getattr(self.evaluator, "actions_for", None)
        return (
            actions_for(self._live_state_key(request_id))
            if actions_for is not None
            else {}
        )

    def policy_score_steps_for(self, request_id: str) -> dict[str, int]:
        self._ensure_device_archived(request_id)
        if request_id in self.archived_policy_score_steps:
            return dict(self.archived_policy_score_steps[request_id])
        score_steps_for = getattr(self.evaluator, "score_steps_for", None)
        return (
            score_steps_for(self._live_state_key(request_id))
            if score_steps_for is not None
            else {}
        )

    def begin_forward(self, num_scheduled_tokens: Any) -> None:
        if self.evaluation_mode == EVALUATION_DEVICE:
            self._begin_forward_device(num_scheduled_tokens)
            return
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
                and self.evaluator.needs_prompt(self._state_key(request.request_id, row))
            )
            max_decode_step = self.evaluator.max_decode_step
            decode_endpoint = (
                self.evaluator.observes_decode
                and step > 0
                and not request.actioned
                and (max_decode_step is None or step <= max_decode_step)
            )
            eligible = self._account_forward(
                request, step, count, bool(prompt_endpoint or decode_endpoint)
            )
            if eligible:
                eligible_rows.append(row)
                eligible_indices.append(final_index)
        self.skip_capture = not eligible_rows
        current_rows = eligible_rows
        final_indices = eligible_indices
        self.forward_epoch += 1
        self.current_rows = current_rows
        self.final_token_indices = final_indices
        self.pending_actions = {}
        self.num_forward_tokens = running
        if self.observation_mode == OBSERVATION_GRAPH:
            from lmsm.integrations.vllm_graph import CAPTURE

            CAPTURE.set_step_epoch(self.forward_epoch)

    def _account_forward(
        self, request: RequestState, step: int, count: int, eligible: bool
    ) -> bool:
        """Record a processed position and return whether to observe it."""

        account = self.accounts.get(request.request_id)
        if account is None:
            account = RequestAccount(request.request_id, len(request.prompt_token_ids))
            self.accounts[request.request_id] = account
            account.fail(step, "processed without an execution account")
        account.mark_processed(step)
        if account.failed:
            # The decision cannot be computed from complete evidence anymore.
            # Release is denied, so the policy is not consulted again.
            return False
        if self.requirements.requires_observation(step, account.decision_step) and not eligible:
            if step == 0 and count != len(request.prompt_token_ids):
                reason = (
                    f"the prompt pass scheduled {count} of "
                    f"{len(request.prompt_token_ids)} prompt tokens (chunked prefill "
                    "or prefix caching), so the required prompt observation was not "
                    "captured"
                )
            else:
                reason = f"the required observation at step {step} was not scheduled"
            account.fail(step, reason)
            return False
        return eligible

    def consume_graph_capture(self) -> None:
        """Deliver this forward's graph-captured rows to the policy (host path).

        Checks the device epoch written by ``lmsm::capture_rows``.  When the
        capture did not run in this forward, nothing is delivered and
        ``apply_actions`` fails the requests that needed the observation.
        """

        if self.consumed_epoch == self.forward_epoch or self.captured_epoch == self.forward_epoch:
            return
        if self.skip_capture:
            self.pending_actions = {}
            self.captured_epoch = self.forward_epoch
            return
        from lmsm.integrations.vllm_graph import CAPTURE, capture_ok

        self.graph_epoch_checks += 1
        if not bool(capture_ok(self.forward_epoch).item()):
            self.graph_epoch_failures += 1
            return
        assert CAPTURE.buffer is not None
        self.capture(CAPTURE.buffer[: self.num_forward_tokens])

    def capture(self, hidden: torch.Tensor) -> None:
        if self.skip_capture:
            if self.captured_epoch == self.forward_epoch:
                raise AssertionError("duplicate activation capture in one forward")
            self.pending_actions = {}
            self.captured_epoch = self.forward_epoch
            return
        live_forward = self.forward_epoch > self.captured_epoch
        if not self.rows and not live_forward:
            self._note_idle_hook_call(hidden)
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

        current_rows = self.current_rows
        final_token_indices = self.final_token_indices
        if self.inject_drop_observations:
            current_rows, final_token_indices = self._inject_dropped_observations(
                current_rows, final_token_indices
            )
            if not current_rows:
                self.pending_actions = {}
                self.captured_epoch = self.forward_epoch
                return

        final_rows = hidden[final_token_indices]
        request_ids = [self.rows[row].request_id for row in current_rows]
        steps = [len(self.rows[row].output_token_ids) for row in current_rows]
        if self.dump_observations:
            self.dumped_observations.append(
                (list(request_ids), list(steps), final_rows.detach().float().cpu().numpy())
            )
        state_keys = (
            [
                self._state_key(request_id, row)
                for request_id, row in zip(request_ids, current_rows, strict=True)
            ]
            if self.inject_row_keyed_state
            else request_ids
        )
        try:
            decisions = self.evaluator.evaluate(state_keys, final_rows, steps)
        except Exception as error:  # noqa: BLE001 - recorded per request
            self.evaluation_errors += 1
            for request_id, step in zip(request_ids, steps, strict=True):
                account = self.accounts.get(request_id)
                if account is not None:
                    account.fail(
                        step,
                        f"policy evaluation raised {type(error).__name__}: {error}",
                    )
            self.pending_actions = {}
            self.captured_epoch = self.forward_epoch
            return
        if len(decisions) != len(current_rows):
            raise AssertionError("policy evaluator returned the wrong number of decisions")
        self._account_evaluations(request_ids, steps, state_keys)
        nonfinite = self._fail_nonfinite(final_rows, request_ids, steps, state_keys)
        self.pending_actions = {
            row: decision
            for position, (row, decision) in enumerate(zip(current_rows, decisions, strict=True))
            if decision.action != "allow"
            and not self.rows[row].actioned
            and position not in nonfinite
        }
        self.captured_epoch = self.forward_epoch

    def _note_idle_hook_call(self, hidden: Any) -> None:
        """Record the hook width seen outside generation (startup profiling)."""

        if _is_compiling():
            return
        if isinstance(hidden, torch.Tensor):
            self.idle_hook_calls += 1
            self.idle_observed_width = int(hidden.shape[-1])

    def _inject_dropped_observations(
        self, rows: list[int], indices: list[int]
    ) -> tuple[list[int], list[int]]:
        """Test only: withhold selected observations from the evaluator."""

        kept_rows, kept_indices = [], []
        for row, index in zip(rows, indices, strict=True):
            request = self.rows[row]
            step = len(request.output_token_ids)
            if step in self.inject_drop_observations.get(request.request_id, ()):
                account = self.accounts.get(request.request_id)
                if account is not None:
                    account.injected_faults.append(
                        {"kind": "drop_observation", "step": step}
                    )
                    account.fail(
                        step,
                        f"the observation at step {step} was dropped by test "
                        "injection (--inject-drop-observation)",
                    )
                continue
            kept_rows.append(row)
            kept_indices.append(index)
        return kept_rows, kept_indices

    def _account_evaluations(
        self, request_ids: list[str], steps: list[int], state_keys: list[str]
    ) -> None:
        for request_id, step, key in zip(request_ids, steps, state_keys, strict=True):
            account = self.accounts.get(request_id)
            if account is None:
                continue
            account.mark_evaluated(step)
            if self.inject_row_keyed_state:
                contributors = self.state_provenance.setdefault(key, set())
                contributors.add(request_id)
                account.state_key = key
                account.foreign_evidence.update(contributors - {request_id})

    def _fail_nonfinite(
        self,
        rows: torch.Tensor,
        request_ids: list[str],
        steps: list[int],
        state_keys: list[str],
    ) -> set[int]:
        """Fail closed the requests whose evidence at this step was not finite.

        A NaN or infinite value in the delivered activation row, or in a pooled
        mean or score the evaluator computed for the decision (reported by its
        ``nonfinite_step_for``), cannot inform a decision.  The request fails
        (``execution_failed``, fixed refusal), and its decision at this step is
        dropped.  A failed request is not evaluated again.  Rows are evaluated
        independently, so the other requests in the batch are unaffected.
        Returns the positions of the failed rows.

        Host evaluation requires synchronous scheduling (``async_scheduling_off``
        at startup).  Every forward evaluated here is therefore consumed, and its
        step is below the final length.  No discarded forward can fail the
        request, and none of its decisions is discarded either.  Compare the
        device path's ``_on_discarded_forward``.
        """

        finite = torch.isfinite(rows).all(dim=-1).tolist()
        reported = getattr(self.evaluator, "nonfinite_step_for", None)
        failed: set[int] = set()
        for position, (request_id, step, key) in enumerate(
            zip(request_ids, steps, state_keys, strict=True)
        ):
            if finite[position] and (reported is None or reported(key) is None):
                continue
            failed.add(position)
            self.nonfinite_failures += 1
            account = self.accounts.get(request_id)
            if account is not None:
                account.fail(step, _nonfinite_reason(step))
        return failed

    def _fail_uncaptured_forward(self) -> None:
        """Account a forward pass in which the activation hook never ran."""

        self.uncaptured_forwards += 1
        for row in self.current_rows:
            request = self.rows.get(row)
            if request is None:
                continue
            account = self.accounts.get(request.request_id)
            if account is not None:
                account.fail(
                    len(request.output_token_ids),
                    "the activation hook did not run in the forward pass (for "
                    "example under compiled or CUDA-graph execution), so no "
                    "observation was delivered",
                )

    def apply_actions(self, logits: torch.Tensor) -> torch.Tensor:
        if self.captured_epoch != self.forward_epoch:
            # The hook never delivered this step's activations.  Every request
            # that required an observation now lacks it, and no row is acted on.
            if self.consumed_epoch != self.forward_epoch:
                self._fail_uncaptured_forward()
            self.pending_actions = {}
            self.consumed_epoch = self.forward_epoch
            return logits
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
            self._account_decision(request, row, len(prefix), decision.action)
            logits[row].fill_(float("-inf"))
            logits[row, self.eos_token_id] = 0.0

        self.pending_actions = {}
        self.consumed_epoch = self.forward_epoch
        return logits

    def _account_decision(
        self, request: RequestState, row: int, step: int, action: str
    ) -> None:
        account = self.accounts.get(request.request_id)
        if account is None:
            return
        account.record_decision(step, action)
        if self.inject_row_keyed_state:
            contributors = self.state_provenance.get(
                self._state_key(request.request_id, row), set()
            )
            account.decision_used_foreign_evidence = bool(
                contributors - {request.request_id}
            )

    def record_for(self, request_id: str) -> ActionRecord | None:
        if self.evaluation_mode == EVALUATION_DEVICE:
            self._flush_device()
        return next(
            (
                record
                for record in reversed(self.action_records)
                if record.request_id == request_id
            ),
            None,
        )

    def execution_record_for(
        self,
        request_id: str,
        generated_token_count: int,
        finish_reason: str | None = None,
    ):
        """Assign the request's execution state and build its record."""

        if self.evaluation_mode == EVALUATION_DEVICE:
            self._materialize_device_account(request_id, generated_token_count)
        account = self.accounts.get(request_id)
        if (
            account is not None
            and self.inject_row_keyed_state
            and account.decision_used_foreign_evidence is None
        ):
            account.decision_used_foreign_evidence = bool(account.foreign_evidence)
        outcome = assign_execution_state(
            self.requirements, account, generated_token_count
        )
        record = build_execution_record(
            self.requirements,
            account,
            outcome,
            request_id=request_id,
            finish_reason=finish_reason,
            association_ablation=self.inject_row_keyed_state,
        )
        if self.inject_row_keyed_state:
            record["association_ablation"]["mode"] = self.row_keyed_mode
        self.execution_records[request_id] = record
        return outcome, record

    def record_engine_violations(self, violations: list[EngineViolation]) -> None:
        self.gate_constructions += 1
        self.engine_violations = list(violations)


def _nonfinite_reason(step: int) -> str:
    """The recorded runtime failure for non-finite evidence (host and device paths)."""

    return (
        f"non-finite policy evidence at step {step}: an observed activation, pooled "
        "mean or score was NaN or infinite, so no decision can be computed from it"
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
        # Execution-contract admission checks.
        n = getattr(sampling_params, "n", 1)
        if n is not None and int(n) != 1:
            raise ValueError("LMSM vLLM requires n=1 (one logits row per request)")
        requirements = STATE.requirements
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if (
            requirements is not None
            and requirements.evaluation_steps
            and max_tokens is not None
            and int(max_tokens) <= max(requirements.evaluation_steps)
        ):
            raise ValueError(
                f"max_tokens={max_tokens} cannot reach {requirements.policy_name}'s "
                f"evaluation point; it requires max_tokens >= "
                f"{max(requirements.evaluation_steps) + 1}"
            )

    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool):
        del is_pin_memory
        violations = check_vllm_config(
            vllm_config,
            observation_mode=STATE.observation_mode,
            allow_async_scheduling=STATE.evaluation_mode == EVALUATION_DEVICE,
            allow_engine_process=STATE.evaluation_mode == EVALUATION_DEVICE,
            allow_partial_prefill=STATE.evaluation_mode == EVALUATION_DEVICE,
        )
        STATE.async_scheduling = bool(
            getattr(vllm_config.scheduler_config, "async_scheduling", False)
        )
        STATE.record_engine_violations(violations)
        if violations and not STATE.allow_unsupported_engine:
            raise UnsupportedEngineError(violations)
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("LMSM vLLM requires TP=1 and PP=1")
        if vllm_config.speculative_config is not None:
            raise ValueError("LMSM vLLM does not support speculative decoding")
        eos = vllm_config.model_config.hf_config.eos_token_id
        STATE.eos_token_id = int(eos[0] if isinstance(eos, list) else eos)
        if STATE.evaluation_mode == EVALUATION_DEVICE:
            STATE.init_device(
                max_rows=int(vllm_config.scheduler_config.max_num_seqs),
                max_steps=int(vllm_config.model_config.max_model_len) + 1,
                device=device,
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Any | None) -> None:
        STATE.update_batch(batch_update)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if STATE.evaluation_mode == EVALUATION_DEVICE:
            return STATE.apply_device(logits)
        if STATE.observation_mode == OBSERVATION_GRAPH:
            STATE.consume_graph_capture()
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
_ORIGINAL_MODEL_FORWARD: Callable[..., Any] | None = None


def install_vllm_integration(
    evaluator: Any,
    fixed_refusal_tokens: list[int],
    *,
    allow_unsupported_engine: bool = False,
    inject_drop_observations: dict[str, Any] | None = None,
    inject_row_keyed_state: bool | str = False,
    observation_mode: str = OBSERVATION_HOOK,
    graph_capture_disabled_for_test: bool = False,
    evaluation_mode: str = EVALUATION_HOST,
) -> None:
    """Install the profile-selected observation and current-input metadata bridge.

    ``observation_mode="hook"`` (default) reads the declared activation with a
    Python forward hook and requires eager execution.  ``"graph"`` places the
    ``lmsm::capture_rows`` operator at the same point, which works with
    torch.compile and CUDA graphs (see ``lmsm.integrations.vllm_graph``).  The
    other keyword arguments are knobs for the mediation tests (see
    ``BatchPolicyState.configure``).
    """

    global _PATCHED, _ORIGINAL_LOAD_MODEL, _ORIGINAL_PREPARE_INPUTS, _ORIGINAL_MODEL_FORWARD
    if _PATCHED:
        raise RuntimeError("LMSM vLLM runtime is already installed")
    STATE.configure(
        evaluator,
        fixed_refusal_tokens,
        allow_unsupported_engine=allow_unsupported_engine,
        inject_drop_observations=inject_drop_observations,
        inject_row_keyed_state=inject_row_keyed_state,
        observation_mode=observation_mode,
        graph_capture_disabled_for_test=graph_capture_disabled_for_test,
        evaluation_mode=evaluation_mode,
    )

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _ORIGINAL_LOAD_MODEL = GPUModelRunner.load_model
    _ORIGINAL_PREPARE_INPUTS = GPUModelRunner._prepare_inputs

    @wraps(_ORIGINAL_LOAD_MODEL)
    def load_model_and_hook(runner: Any, *args: Any, **kwargs: Any) -> None:
        assert _ORIGINAL_LOAD_MODEL is not None
        _ORIGINAL_LOAD_MODEL(runner, *args, **kwargs)
        STATE.runner = runner
        STATE.model_runner_class = f"{type(runner).__module__}.{type(runner).__qualname__}"
        activation_key = STATE.activation_key
        if activation_key is None:
            raise AssertionError("LMSM activation key is not configured")
        from lmsm.integrations.vllm_graph import (
            POINT_MODULE_INPUT,
            install_graph_capture,
            resolve_observation_point,
        )

        point, module_name = resolve_observation_point(runner.model, activation_key)
        STATE.resolved_module = module_name
        STATE.observation_point = point
        model_width = _model_hidden_width(runner.model_config.hf_config)
        if STATE.expected_hidden_width is None:
            STATE.expected_hidden_width = model_width
        if STATE.observation_mode == OBSERVATION_GRAPH:
            install_graph_capture(
                runner.model,
                module_name,
                max_tokens=_max_forward_tokens(runner),
                width=int(STATE.expected_hidden_width or model_width),
                dtype=runner.model_config.dtype,
                device=runner.device,
                disabled_for_test=STATE.graph_capture_disabled_for_test,
                point=point,
            )
            STATE.hook_name = module_name
            return
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
        if point == POINT_MODULE_INPUT:
            module.register_forward_pre_hook(_capture_module_input)
        else:
            module.register_forward_hook(_capture_residual_stream_output)

    @wraps(_ORIGINAL_PREPARE_INPUTS)
    def prepare_inputs(runner: Any, scheduler_output: Any, num_scheduled_tokens: Any):
        assert _ORIGINAL_PREPARE_INPUTS is not None
        prepared = _ORIGINAL_PREPARE_INPUTS(
            runner, scheduler_output, num_scheduled_tokens
        )
        # This array is in the current post-update InputBatch row order.  Its
        # values define the packed forward layout.  No scheduler dictionary or
        # internal request ID participates in the join.
        STATE.logits_indices = prepared[0] if isinstance(prepared, tuple) else None
        STATE.begin_forward(num_scheduled_tokens)
        return prepared

    GPUModelRunner.load_model = load_model_and_hook
    GPUModelRunner._prepare_inputs = prepare_inputs
    from vllm.v1.worker.gpu_worker import Worker

    # Inherited by vLLM's engine-core process when it is forked.
    Worker.lmsm_rpc = _worker_rpc
    if STATE.evaluation_mode == EVALUATION_DEVICE:
        _ORIGINAL_MODEL_FORWARD = GPUModelRunner._model_forward

        @wraps(_ORIGINAL_MODEL_FORWARD)
        def model_forward(runner: Any, *args: Any, **kwargs: Any):
            assert _ORIGINAL_MODEL_FORWARD is not None
            output = _ORIGINAL_MODEL_FORWARD(runner, *args, **kwargs)
            STATE.after_forward()
            return output

        GPUModelRunner._model_forward = model_forward
    _PATCHED = True


GRAPH_CAPTURE_SIGNATURE_VERSION = 1


def engine_kwargs_for_observation() -> dict[str, Any]:
    """Extra ``vllm.LLM`` arguments the installed observation needs.

    For graph observation this returns ``additional_config`` carrying the
    capture signature.  vLLM hashes ``additional_config`` into its
    torch.compile and AOT cache keys.  A compiled graph built without the
    capture op, for example by an unmodified vLLM or by the test knob that
    omits the op, is therefore never reused for a graph-observation run.  The startup
    check still verifies that the op ran.
    """

    if STATE.observation_mode != OBSERVATION_GRAPH:
        return {}
    module = _split_hook_target(STATE.activation_key)[1] if STATE.activation_key else None
    signature = {
        "op": "lmsm::capture_rows",
        "version": GRAPH_CAPTURE_SIGNATURE_VERSION,
        "module": module,
        "capture_disabled_for_test": STATE.graph_capture_disabled_for_test,
    }
    if STATE.activation_key and not STATE.activation_key.startswith("pre:"):
        # Residual stream after a decoder layer.  The signature names its op and
        # its observation point.
        from lmsm.integrations.vllm_graph import POINT_RESIDUAL_STREAM

        signature["op"] = "lmsm::capture_residual_sum"
        signature["point"] = POINT_RESIDUAL_STREAM
    return {"additional_config": {"lmsm_graph_observation": signature}}


def uninstall_vllm_integration() -> None:
    global _PATCHED, _ORIGINAL_LOAD_MODEL, _ORIGINAL_PREPARE_INPUTS, _ORIGINAL_MODEL_FORWARD
    if not _PATCHED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    GPUModelRunner.load_model = _ORIGINAL_LOAD_MODEL
    GPUModelRunner._prepare_inputs = _ORIGINAL_PREPARE_INPUTS
    if _ORIGINAL_MODEL_FORWARD is not None:
        GPUModelRunner._model_forward = _ORIGINAL_MODEL_FORWARD
        _ORIGINAL_MODEL_FORWARD = None
    _ORIGINAL_LOAD_MODEL = None
    _ORIGINAL_PREPARE_INPUTS = None
    _PATCHED = False


def verify_engine(
    llm: Any | None = None,
    *,
    earlier_violations: list[EngineViolation] | tuple[EngineViolation, ...] = (),
) -> dict[str, Any]:
    """Complete the startup check after the engine has been constructed.

    Combines the resolved-configuration check made inside the worker with the
    installation checks that need a loaded model.  The gate must share this
    process's state, the hook must sit on the declared module of Model Runner
    V1, and the width that the hook observed during startup profiling must
    match the backend.  Raises ``UnsupportedEngineError`` unless the test-only
    ``allow_unsupported_engine`` knob is set.
    """

    stage = "engine_installation"
    facts = worker_call(llm, "_worker_engine_facts")
    remote = STATE.gate_constructions == 0 and facts.get("gate_constructions", 0) > 0
    if remote:
        # Engine core in vLLM's background process (device evaluation only).
        # Take the worker's view of the installation.  The release step reads
        # the worker's records through collective_rpc.
        _adopt_worker_facts(facts)
    violations = list(earlier_violations) + list(STATE.engine_violations)
    if STATE.gate_constructions == 0:
        violations.append(
            EngineViolation(
                "in_process_engine_core",
                "the LMSM enforcement gate was never constructed in this process, "
                "so the engine core does not share the release step's state",
                stage,
            )
        )
    client = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if client is not None and type(client).__name__ != "InprocClient" and not remote:
        violations.append(
            EngineViolation(
                "in_process_engine_core",
                f"the engine core client is {type(client).__name__}, not InprocClient",
                stage,
            )
        )
    if STATE.model_runner_class != V1_MODEL_RUNNER:
        violations.append(
            EngineViolation(
                "model_runner_v1",
                f"model runner {STATE.model_runner_class!r} loaded the model; the "
                f"integration requires {V1_MODEL_RUNNER}",
                stage,
            )
        )
    declared = (
        _split_hook_target(STATE.activation_key)[1]
        if STATE.activation_key is not None
        else None
    )
    if STATE.hook_name is None or STATE.hook_name != STATE.resolved_module:
        violations.append(
            EngineViolation(
                "hook_installed",
                f"the backend declares module {declared!r} (vLLM module "
                f"{STATE.resolved_module!r}); the hook is installed on "
                f"{STATE.hook_name!r}",
                stage,
            )
        )
    graph_check: dict[str, Any] | None = None
    if STATE.observation_mode == OBSERVATION_GRAPH:
        graph_check = _verify_graph_capture(llm, stage, violations)
    elif STATE.idle_observed_width is None:
        violations.append(
            EngineViolation(
                "hook_width",
                "the activation hook did not run during startup profiling, so its "
                f"width could not be checked against the backend's "
                f"{STATE.expected_hidden_width}",
                stage,
            )
        )
    elif (
        STATE.expected_hidden_width is not None
        and STATE.idle_observed_width != STATE.expected_hidden_width
    ):
        violations.append(
            EngineViolation(
                "hook_width",
                f"the hook observed width {STATE.idle_observed_width}; the backend "
                f"expects {STATE.expected_hidden_width}",
                stage,
            )
        )

    if not violations:
        status = "passed"
    elif STATE.allow_unsupported_engine:
        status = "bypassed"
    else:
        status = "refused"
    STATE.engine_check = {
        "status": status,
        "allow_unsupported_engine": STATE.allow_unsupported_engine,
        "violations": [violation.to_dict() for violation in violations],
        "violated_requirements": sorted({violation.requirement for violation in violations}),
        "model_runner": STATE.model_runner_class,
        "hook_module": STATE.hook_name,
        "observed_hook_width": STATE.idle_observed_width,
        "backend_width": STATE.expected_hidden_width,
        "gate_constructions": STATE.gate_constructions,
        "observation_mode": STATE.observation_mode,
    }
    if graph_check is not None:
        STATE.engine_check["graph_capture"] = graph_check
    if violations and not STATE.allow_unsupported_engine:
        raise UnsupportedEngineError(violations)
    return dict(STATE.engine_check)


def _worker_engine_facts() -> dict[str, Any]:
    """Installation facts that live in the worker process (RPC target)."""

    from lmsm.integrations.vllm_graph import CAPTURE

    return {
        "gate_constructions": STATE.gate_constructions,
        "engine_violations": [v.to_dict() for v in STATE.engine_violations],
        "model_runner_class": STATE.model_runner_class,
        "hook_name": STATE.hook_name,
        "resolved_module": STATE.resolved_module,
        "observation_point": STATE.observation_point,
        "idle_observed_width": STATE.idle_observed_width,
        "idle_hook_calls": STATE.idle_hook_calls,
        "expected_hidden_width": STATE.expected_hidden_width,
        "async_scheduling": STATE.async_scheduling,
        "capture": {
            "installed": CAPTURE.installed,
            "module_name": CAPTURE.module_name,
            "point": CAPTURE.point,
            "impl_calls": CAPTURE.impl_calls,
            "observed_width": CAPTURE.observed_width,
            "max_tokens": CAPTURE.max_tokens,
            "max_rows_seen": CAPTURE.max_rows_seen,
            "disabled_for_test": CAPTURE.disabled_for_test,
        },
        "pid": os.getpid(),
    }


def _adopt_worker_facts(facts: dict[str, Any]) -> None:
    from lmsm.execution_contract import EngineViolation as Violation
    from lmsm.integrations.vllm_graph import CAPTURE

    STATE.gate_constructions = facts["gate_constructions"]
    STATE.engine_violations = [
        Violation(v["requirement"], v["detail"], v.get("stage", "")) for v in facts["engine_violations"]
    ]
    STATE.model_runner_class = facts["model_runner_class"]
    STATE.hook_name = facts["hook_name"]
    STATE.resolved_module = facts["resolved_module"]
    STATE.observation_point = facts["observation_point"]
    STATE.idle_observed_width = facts["idle_observed_width"]
    STATE.idle_hook_calls = facts["idle_hook_calls"]
    STATE.expected_hidden_width = facts["expected_hidden_width"]
    STATE.async_scheduling = facts["async_scheduling"]
    STATE.remote_worker_pid = facts["pid"]
    capture = facts["capture"]
    CAPTURE.module_name = capture["module_name"]
    CAPTURE.point = capture["point"]
    CAPTURE.impl_calls = capture["impl_calls"]
    CAPTURE.observed_width = capture["observed_width"]
    CAPTURE.max_tokens = capture["max_tokens"]
    CAPTURE.max_rows_seen = capture["max_rows_seen"]
    CAPTURE.disabled_for_test = capture["disabled_for_test"]
    CAPTURE.remote_installed = capture["installed"]


def worker_call(llm: Any | None, name: str, *args: Any) -> Any:
    """Call a runtime function where the worker state lives.

    In-process engines share this module's state.  When vLLM runs the engine
    core in its background process (device evaluation only), the call goes
    through ``collective_rpc`` and runs in the worker.
    """

    if llm is None or STATE.gate_constructions > 0 and STATE.remote_worker_pid is None:
        return globals()[name](*args)
    # By method name: vLLM serializes RPC arguments with msgspec and refuses
    # callables, so the worker class carries an ``lmsm_rpc`` method.
    results = llm.collective_rpc(RPC_METHOD, args=(name, list(args)))
    return results[0]


RPC_METHOD = "lmsm_rpc"
RPC_FUNCTIONS = (
    "_worker_engine_facts",
    "release_views",
    "runtime_counters",
    "save_observation_dump",
)


def _worker_rpc(_worker: Any, name: str, args: list) -> Any:
    if name not in RPC_FUNCTIONS:
        raise ValueError(f"not an LMSM worker function: {name!r}")
    return globals()[name](*args)


def release_views(items: list[tuple[str, list[int], str | None]]) -> list[dict[str, Any]]:
    """Release every finished request and return what the runner records.

    Runs where the worker state lives (see ``worker_call``).
    """

    views = []
    for request_id, ordinary_tokens, finish_reason in items:
        view = buffered_output(request_id, list(ordinary_tokens), finish_reason=finish_reason)
        views.append(
            {
                "view": view,
                "feature_maxima": feature_maxima(request_id),
                "policy_scores": policy_scores(request_id),
                "policy_score_steps": policy_score_steps(request_id),
                "policy_actions": policy_actions(request_id),
            }
        )
    return views


def _verify_graph_capture(
    llm: Any | None, stage: str, violations: list[EngineViolation]
) -> dict[str, Any]:
    """Startup evidence that the capture op is part of the compiled model.

    The op's Python implementation runs during startup profiling, warmup, and
    CUDA-graph capture (never during replay), so a nonzero call count with the
    backend's width shows that the compiled model contains the op on the
    declared module.  The per-forward epoch check covers every later step.
    """

    from lmsm.integrations.vllm_graph import CAPTURE

    installed = CAPTURE.installed or getattr(CAPTURE, "remote_installed", False)
    if not installed or CAPTURE.module_name != STATE.hook_name:
        violations.append(
            EngineViolation(
                "graph_capture_installed",
                f"the capture op is installed on {CAPTURE.module_name!r}; the backend "
                f"declares {STATE.hook_name!r}",
                stage,
            )
        )
    if CAPTURE.impl_calls == 0:
        violations.append(
            EngineViolation(
                "graph_capture_ran",
                "the capture op never ran during startup profiling or graph capture, "
                "so the compiled model does not contain it",
                stage,
            )
        )
    elif (
        STATE.expected_hidden_width is not None
        and CAPTURE.observed_width != STATE.expected_hidden_width
    ):
        violations.append(
            EngineViolation(
                "graph_capture_width",
                f"the capture op observed width {CAPTURE.observed_width}; the backend "
                f"expects {STATE.expected_hidden_width}",
                stage,
            )
        )
    compilation: dict[str, Any] = {}
    try:
        config = llm.llm_engine.vllm_config  # type: ignore[union-attr]
        cc = config.compilation_config
        compilation = {
            "mode": str(getattr(cc.mode, "name", cc.mode)),
            "cudagraph_mode": str(getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode)),
            "enforce_eager": bool(config.model_config.enforce_eager),
            "max_cudagraph_capture_size": getattr(cc, "max_cudagraph_capture_size", None),
        }
    except AttributeError:
        pass
    return {
        "module": CAPTURE.module_name,
        "point": CAPTURE.point,
        "startup_impl_calls": CAPTURE.impl_calls,
        "observed_width": CAPTURE.observed_width,
        "buffer_rows": CAPTURE.max_tokens,
        "max_rows_seen": CAPTURE.max_rows_seen,
        "disabled_for_test": CAPTURE.disabled_for_test,
        "compilation": compilation,
    }


def save_observation_dump(path: str) -> int:
    """Test only: write the rows recorded under ``LMSM_DUMP_OBSERVATIONS``."""

    ids: list[str] = []
    steps: list[int] = []
    rows = []
    for request_ids, request_steps, values in STATE.dumped_observations:
        ids.extend(request_ids)
        steps.extend(request_steps)
        rows.append(values)
    matrix = np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(
        path, request_ids=np.asarray(ids), steps=np.asarray(steps, dtype=np.int64), rows=matrix
    )
    return len(ids)


def execution_requirements() -> dict[str, Any] | None:
    """Return the active policy's declared execution requirements."""

    return STATE.requirements.to_dict() if STATE.requirements is not None else None


def runtime_counters() -> dict[str, int]:
    """Return contract counters that are not attached to one request."""

    counters = {
        "evaluation_errors": STATE.evaluation_errors,
        "nonfinite_failures": STATE.nonfinite_failures,
        "uncaptured_forwards": STATE.uncaptured_forwards,
        "startup_hook_calls": STATE.idle_hook_calls,
    }
    if STATE.observation_mode == OBSERVATION_GRAPH:
        from lmsm.integrations.vllm_graph import CAPTURE

        counters.update(
            {
                "graph_epoch_checks": STATE.graph_epoch_checks,
                "graph_epoch_failures": STATE.graph_epoch_failures,
                "graph_capture_impl_calls": CAPTURE.impl_calls,
                "evaluation_mode": STATE.evaluation_mode,
                "device_steps": STATE.device_steps,
                "device_flushes": STATE.device_flushes,
                "async_scheduling": STATE.async_scheduling,
                "discarded_async_decisions": STATE.discarded_async_decisions,
                "discarded_async_nonfinite": STATE.discarded_async_nonfinite,
            }
        )
    if STATE.profile_steps:
        totals = dict(STATE.profile_totals)
        if STATE.profile_events:
            torch.cuda.synchronize()
            totals["gpu_apply_ms"] = sum(
                start.elapsed_time(end) for start, end in STATE.profile_events
            )
            totals["gpu_apply_calls"] = len(STATE.profile_events)
        counters["profile"] = totals
    return counters


def _capture_module_input(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
    STATE.capture(args[0])


def _capture_residual_stream_output(
    _module: torch.nn.Module, _args: tuple[Any, ...], output: Any
) -> None:
    """Residual stream after a vLLM decoder layer that returns (hidden, residual).

    The same sum, in the same dtype, as ``lmsm::capture_residual_sum`` and as
    the Hugging Face layer output (``residual + hidden_states``).
    """

    hidden, residual = output
    STATE.capture(hidden + residual)


def _split_hook_target(activation_key: str) -> tuple[str, str]:
    if activation_key.startswith("pre:"):
        return "pre", activation_key[len("pre:") :]
    return "post", activation_key


def _max_forward_tokens(runner: Any) -> int:
    """Largest packed token count of any forward (real, profiling, or capture)."""

    candidates = [
        getattr(runner, "max_num_tokens", None),
        getattr(getattr(runner, "scheduler_config", None), "max_num_batched_tokens", None),
    ]
    values = [int(value) for value in candidates if value]
    if not values:
        raise AssertionError("cannot determine the maximum tokens per forward")
    return max(values)


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


def buffered_output(
    request_id: str,
    ordinary_output_tokens: list[int],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Release one finished request according to its execution state.

    Completed requests, and not-reached requests whose policy declares
    ``allow_with_record``, receive the action record's release when the policy
    acted and the buffered model output otherwise.  Failed requests, and
    not-reached requests under ``deny``, receive the operator's fixed refusal.  Every view carries an ``execution_record``.
    """

    STATE.finalize_device_request(request_id, ordinary_output_tokens)
    record = STATE.record_for(request_id)
    outcome, execution_record = STATE.execution_record_for(
        request_id, len(ordinary_output_tokens), finish_reason
    )
    released_as_executed = outcome.state == COMPLETED or (
        outcome.state == NOT_REACHED and outcome.disposition == ALLOW_WITH_RECORD
    )
    if released_as_executed:
        if record is None:
            view = {
                "request_id": request_id,
                "ordinary_model_prefix_tokens": list(ordinary_output_tokens),
                "forced_termination_token": None,
                "fixed_refusal_tokens": [],
                "admitted_prefix_tokens": list(ordinary_output_tokens),
                "released_output_tokens": list(ordinary_output_tokens),
                "action_record": None,
            }
        else:
            view = {**record.to_dict(), "action_record": record.to_dict()}
        view["execution_record"] = execution_record
        return view

    refusal_tokens = list(STATE.fixed_refusal_tokens)
    return {
        "request_id": request_id,
        "ordinary_model_prefix_tokens": list(ordinary_output_tokens),
        "forced_termination_token": None,
        "fixed_refusal_tokens": refusal_tokens,
        "admitted_prefix_tokens": [],
        "released_output_tokens": refusal_tokens,
        "action_record": record.to_dict() if record is not None else None,
        "execution_record": execution_record,
    }


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
