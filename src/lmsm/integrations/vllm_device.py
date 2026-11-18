"""Device-resident policy evaluation for the graph-observation vLLM path.

The host evaluators in ``lmsm.policy_evaluator`` copy activation rows to the
CPU and evaluate the probes with numpy, which needs a host synchronization on
every step.  The engines here keep the same per-request policy state on the
GPU and run the same arithmetic there.  A step then needs no synchronization
beyond vLLM's own.

* Running sums are float32 and are accumulated in the same order as the numpy
  code.  The first observation is added to zero, which is exact.  Pooled means
  are float32 divisions, so they are bitwise identical to the host path for
  identical activations.
* Probe scores are float64 and are computed from the pooled means.  The host
  path computes ``((h - mean) / scale) @ coef + intercept``.  The device path
  uses the algebraically equal ``h @ (coef / scale) + (intercept - mean / scale
  @ coef)``.  Both round at the 1e-15 relative level, far below any threshold
  margin.
* First crossings, rule selection in active-rule order, and the decision step
  follow the host evaluators exactly.

State is indexed by a slot.  The runtime assigns a slot to the trusted request
ID on admission and frees it when the request leaves the batch, so batch-row
moves never move evidence between requests.  For each slot and processed step,
every engine records whether the observation was delivered, as gated by the
device epoch check.  The execution contract reads this record when the request
finishes.  Each engine also records, per slot, the first step at which an
observed activation, a pooled mean or a score the policy uses was NaN or
infinite (``nonfinite``).  This check runs next to the arithmetic and does not
change it.  From that step on, the slot takes no action, and the request fails
closed when it finishes.  The exception is a step on a forward that vLLM
discarded (see ``BatchPolicyState``).

All per-step inputs live in one static index buffer (rows, slots, steps, token
positions, validity, expected epoch), padded to the maximum batch width.
Padding rows point at a dummy slot and a dummy batch row.  Each step function
therefore uses fixed shapes and fixed addresses.  It is captured once as a CUDA
graph and replayed every step, which takes one launch instead of about 30
small kernels.  ``LMSM_DEVICE_GRAPHS=0`` runs the same functions without
graphs.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from lmsm.policy_evaluator import (
    AnytimePolicyEvaluator,
    CheckpointPolicyEvaluator,
    SharedFeaturePolicyEvaluator,
)


@dataclass
class StepFlags:
    """Host-known facts about one step's eligible rows (no device reads)."""

    any_prompt: bool = False
    any_window_1_end: bool = False
    any_evaluation: bool = False


ROW, SLOT, STEP, TOKEN, VALID, EPOCH = range(6)


def _probe_matrices(probes: list[Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold standardization into a float64 weight matrix and bias."""

    weights = []
    biases = []
    for probe in probes:
        mean = np.asarray(probe.mean, dtype=np.float64)
        scale = np.asarray(probe.scale, dtype=np.float64)
        coef = np.asarray(probe.coef, dtype=np.float64)
        weights.append(coef / scale)
        biases.append(np.float64(probe.intercept) - float(np.dot(mean / scale, coef)))
    weight = torch.tensor(np.stack(weights, axis=1), dtype=torch.float64, device=device)
    bias = torch.tensor(np.asarray(biases, dtype=np.float64), dtype=torch.float64, device=device)
    return weight, bias


class _DeviceEngine:
    """Static step buffers, slot bookkeeping, and the delivered-observation map."""

    observes_prompt = False
    observes_decode = True
    max_decode_step: int | None = None

    def __init__(
        self,
        evaluator: Any,
        num_slots: int,
        max_rows: int,
        max_steps: int,
        device: torch.device,
    ):
        self.evaluator = evaluator
        self.profile = evaluator.profile
        self.actions_enabled = bool(
            getattr(
                evaluator,
                "actions_enabled",
                getattr(getattr(evaluator, "decider", None), "actions_enabled", True),
            )
        )
        self.num_slots = int(num_slots)
        self.dummy_slot = self.num_slots
        self.max_rows = int(max_rows)
        self.dummy_row = self.max_rows
        self.max_steps = int(max_steps)
        self.device = device
        self.use_graphs = os.environ.get("LMSM_DEVICE_GRAPHS", "1") != "0"
        self.use_triton = os.environ.get("LMSM_DEVICE_TRITON", "1") != "0"
        if self.use_triton:
            try:
                from lmsm.integrations import vllm_device_kernels  # noqa: F401
            except ImportError:  # pragma: no cover - Triton missing
                self.use_triton = False
        self.instrument = False
        pin = torch.cuda.is_available()
        self.stage = torch.zeros((6, self.max_rows), dtype=torch.int64, pin_memory=pin)
        self.stage_np = self.stage.numpy()
        self.index = torch.zeros((6, self.max_rows), dtype=torch.int64, device=device)
        self.observed = torch.zeros(
            (self.num_slots + 1, self.max_steps), dtype=torch.uint8, device=device
        )
        # First step with non-finite evidence or scores, per slot (-1: none).
        self.nonfinite = torch.full(
            (self.num_slots + 1,), -1, dtype=torch.int64, device=device
        )
        self.ok = torch.zeros((1,), dtype=torch.bool, device=device)
        self.result = torch.zeros((self.max_rows, 6), dtype=torch.float64, device=device)
        self.act_rows = torch.zeros((self.max_rows + 1,), dtype=torch.bool, device=device)
        self.pinned_ok = torch.zeros((1,), dtype=torch.bool, pin_memory=pin)
        self.pinned_result = torch.zeros((self.max_rows, 6), dtype=torch.float64, pin_memory=pin)
        self.graphs: dict[str, Any] = {}
        self.target_ids: list[str] = []
        self.capture_buffer: torch.Tensor | None = None
        self.captured_epoch: torch.Tensor | None = None
        self.last_hidden: torch.Tensor | None = None
        # Hidden width that the step kernels read per capture-buffer row.  The
        # dense-probe engines set it, and their kernels mask every access at it.
        self.width: int | None = None
        self._pad_stage()

    # -- inputs ----------------------------------------------------------
    def bind_capture(self, buffer: torch.Tensor, captured_epoch: torch.Tensor) -> None:
        if self.width is not None and int(buffer.shape[1]) != self.width:
            # The kernels read `width` values from each buffer row.  A wider
            # policy would read the next token's row, which may hold another
            # request's activation.
            raise ValueError(
                f"the capture buffer holds rows of width {int(buffer.shape[1])}; the "
                f"device policy reads width {self.width}"
            )
        self.capture_buffer = buffer
        self.captured_epoch = captured_epoch

    def _pad_stage(self, start: int = 0) -> None:
        stage = self.stage_np
        stage[ROW, start:] = self.dummy_row
        stage[SLOT, start:] = self.dummy_slot
        stage[STEP, start:] = 0
        stage[TOKEN, start:] = 0
        stage[VALID, start:] = 0

    def load(
        self,
        rows: np.ndarray,
        slots: np.ndarray,
        steps: np.ndarray,
        tokens: np.ndarray,
        epoch: int,
    ) -> None:
        """Stage one step's eligible rows (host) and copy them to the device."""

        m = int(rows.shape[0])
        self.loaded_rows = m
        stage = self.stage_np
        stage[ROW, :m] = rows
        stage[SLOT, :m] = slots
        stage[STEP, :m] = steps
        stage[TOKEN, :m] = tokens
        stage[VALID, :m] = 1
        self._pad_stage(m)
        stage[EPOCH, 0] = epoch
        self.index.copy_(self.stage, non_blocking=True)

    def _instrument_rows(self) -> None:
        if self.instrument:
            if self.last_hidden is None:
                raise AssertionError("instrumentation buffer missing")
            self.last_hidden.copy_(self.capture_buffer.index_select(0, self.index[TOKEN]))

    def _inputs(self) -> tuple[torch.Tensor, ...]:
        index = self.index
        rows, slots, steps, tokens = index[ROW], index[SLOT], index[STEP], index[TOKEN]
        valid = (index[VALID] > 0) & self.ok
        return rows, slots, steps, tokens, valid

    def _prologue(self) -> torch.Tensor:
        """Epoch check and delivered-observation map.  Returns float32 rows."""

        assert self.capture_buffer is not None and self.captured_epoch is not None
        index = self.index
        self.ok.copy_(self.captured_epoch == index[EPOCH, 0:1])
        rows, slots, steps, tokens, valid = self._inputs()
        hidden = self.capture_buffer.index_select(0, tokens)
        if self.instrument:
            if self.last_hidden is None:
                raise AssertionError("instrumentation buffer missing")
            self.last_hidden.copy_(hidden)
        self.observed.index_put_(
            (slots, steps.clamp(0, self.max_steps - 1)), valid.to(torch.uint8)
        )
        self.pinned_ok.copy_(self.ok, non_blocking=True)
        return hidden.to(torch.float32)

    def _record_nonfinite(
        self, slots: torch.Tensor, steps: torch.Tensor, bad: torch.Tensor
    ) -> torch.Tensor:
        """Record each slot's first step with non-finite evidence and return which have one.

        ``bad`` marks the rows whose evidence or scores at this step were NaN
        or infinite.  The returned mask (this step or earlier) suppresses the
        slot's actions.  The account fails when the request finishes.
        """

        first = self.nonfinite.index_select(0, slots)
        first = torch.where(bad & (first < 0), steps, first)
        self.nonfinite.index_copy_(0, slots, first)
        return first >= 0

    # -- execution ---------------------------------------------------------
    def _run(self, name: str, function: Callable[[], None]) -> None:
        if not self.use_graphs:
            function()
            return
        graph = self.graphs.get(name)
        if graph is None:
            raise AssertionError(f"device step {name!r} was not captured")
        graph.replay()

    def capture_graphs(self) -> None:
        """Capture every step function as a CUDA graph (once, padding inputs)."""

        if not self.use_graphs or self.graphs:
            return
        if self.instrument and self.last_hidden is None:
            assert self.capture_buffer is not None
            self.last_hidden = torch.zeros(
                (self.max_rows, self.capture_buffer.shape[1]),
                dtype=self.capture_buffer.dtype,
                device=self.device,
            )
        # Padding-only inputs: every write lands in the dummy slot or row.
        self._pad_stage(0)
        self.stage_np[EPOCH, 0] = -2
        self.index.copy_(self.stage)
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                for function in self._step_functions().values():
                    function()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for name, function in self._step_functions().items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                function()
            self.graphs[name] = graph
        torch.cuda.synchronize()

    def _step_functions(self) -> dict[str, Callable[[], None]]:
        raise NotImplementedError

    def run(self, flags: StepFlags) -> bool:
        """Run one step and return whether ``act_rows`` and results were produced."""

        raise NotImplementedError

    # -- slot lifecycle --------------------------------------------------
    def reset_slots(self, slots: torch.Tensor) -> None:
        self.observed.index_fill_(0, slots, 0)
        self.nonfinite.index_fill_(0, slots, -1)
        self._reset_policy_slots(slots)

    def _reset_policy_slots(self, slots: torch.Tensor) -> None:
        raise NotImplementedError

    def archive(self, slots: list[int], lengths: list[int]) -> list[dict[str, Any]]:
        """Copy the listed slots' state to the host (synchronizes)."""

        if not slots:
            return []
        index = torch.tensor(slots, dtype=torch.int64, device=self.device)
        width = max(1, min(self.max_steps, max(lengths) + 1))
        observed = self.observed.index_select(0, index)[:, :width].cpu().numpy()
        nonfinite = self.nonfinite.index_select(0, index).cpu().numpy()
        policy = self._archive_policy(index)
        records = []
        for position, length in enumerate(lengths):
            record = {
                "observed": observed[position, : max(1, min(width, length + 1))].copy(),
                "nonfinite_step": (
                    int(nonfinite[position]) if nonfinite[position] >= 0 else None
                ),
            }
            record.update(policy[position])
            records.append(record)
        return records

    def _archive_policy(self, index: torch.Tensor) -> list[dict[str, Any]]:
        raise NotImplementedError

    def slot_is_fresh(self, slot: int) -> bool:
        """Test only: True if the slot holds no evidence (synchronizes)."""

        index = torch.tensor([slot], dtype=torch.int64, device=self.device)
        if bool(self.observed.index_select(0, index).any().item()):
            return False
        if bool((self.nonfinite.index_select(0, index) >= 0).any().item()):
            return False
        return self._policy_slot_is_fresh(index)

    def _policy_slot_is_fresh(self, index: torch.Tensor) -> bool:
        raise NotImplementedError


class DeviceAnytimeEngine(_DeviceEngine):
    """GPU form of ``AnytimePolicyEvaluator`` (running-prefix mean, first crossing)."""

    def __init__(self, evaluator: AnytimePolicyEvaluator, num_slots, max_rows, max_steps, device):
        super().__init__(evaluator, num_slots, max_rows, max_steps, device)
        probes = list(evaluator.probes)
        self.target_ids = [probe.target_id for probe in probes]
        self.weight, self.bias = _probe_matrices(probes, device)
        self.thresholds = torch.tensor(
            [float(probe.threshold) for probe in probes], dtype=torch.float64, device=device
        )
        width = self.weight.shape[0]
        self.width = int(width)
        count = len(probes)
        slots = self.num_slots + 1
        self.sums = torch.zeros((slots, width), dtype=torch.float32, device=device)
        self.counts = torch.zeros((slots,), dtype=torch.float32, device=device)
        self.maxima = torch.full((slots, count), float("-inf"), dtype=torch.float64, device=device)
        self.max_steps_tensor = torch.full((slots, count), -1, dtype=torch.int64, device=device)
        self.crossed = torch.zeros((slots, count), dtype=torch.bool, device=device)
        self.last_scores = torch.zeros((self.max_rows, count), dtype=torch.float64, device=device)

    def _reset_policy_slots(self, slots):
        self.sums.index_fill_(0, slots, 0.0)
        self.counts.index_fill_(0, slots, 0.0)
        self.maxima.index_fill_(0, slots, float("-inf"))
        self.max_steps_tensor.index_fill_(0, slots, -1)
        self.crossed.index_fill_(0, slots, False)

    def _policy_slot_is_fresh(self, index):
        return bool(
            (self.sums.index_select(0, index) == 0).all().item()
            and (self.counts.index_select(0, index) == 0).all().item()
            and torch.isneginf(self.maxima.index_select(0, index)).all().item()
            and not self.crossed.index_select(0, index).any().item()
        )

    def _step(self) -> None:
        if self.use_triton:
            self._step_triton()
            return
        hidden = self._prologue()
        rows, slots, steps, _tokens, valid = self._inputs()
        weight = valid.to(torch.float32)
        self.sums.index_add_(0, slots, hidden * weight.unsqueeze(1))
        self.counts.index_add_(0, slots, weight)
        pooled = self.sums.index_select(0, slots) / self.counts.index_select(0, slots).clamp_min(
            1.0
        ).unsqueeze(1)
        scores = torch.addmm(self.bias, pooled.to(torch.float64), self.weight)
        self.last_scores.copy_(scores)
        failed = self._record_nonfinite(
            slots,
            steps,
            valid & ~(torch.isfinite(pooled).all(dim=1) & torch.isfinite(scores).all(dim=1)),
        )
        previous = self.maxima.index_select(0, slots)
        update = (scores > previous) & valid.unsqueeze(1)
        maxima = torch.where(update, scores, previous)
        self.maxima.index_copy_(0, slots, maxima)
        self.max_steps_tensor.index_copy_(
            0,
            slots,
            torch.where(update, steps.unsqueeze(1), self.max_steps_tensor.index_select(0, slots)),
        )
        previous_crossed = self.crossed.index_select(0, slots)
        crossed = maxima >= self.thresholds
        newly = crossed & ~previous_crossed
        self.crossed.index_copy_(0, slots, crossed)
        if not self.actions_enabled:
            return
        triggered = newly.any(dim=1) & valid & ~failed
        rule = torch.argmax(newly.to(torch.int32), dim=1)
        self.result[:, 0] = triggered.to(torch.float64)
        self.result[:, 1] = rule.to(torch.float64)
        self.result[:, 2] = maxima.gather(1, rule.unsqueeze(1)).squeeze(1)
        self.act_rows.fill_(False)
        self.act_rows.index_copy_(0, rows, triggered)
        self.pinned_result.copy_(self.result, non_blocking=True)

    def _step_triton(self) -> None:
        from lmsm.integrations.vllm_device_kernels import anytime_step_kernel

        self._instrument_rows()
        if self.actions_enabled:
            self.act_rows.fill_(False)
        width = self.weight.shape[0]
        count = self.weight.shape[1]
        anytime_step_kernel[(self.max_rows,)](
            self.index,
            self.max_rows,
            self.captured_epoch,
            self.capture_buffer,
            self.capture_buffer.stride(0),
            self.observed,
            self.observed.stride(0),
            self.max_steps,
            self.sums,
            self.counts,
            self.weight,
            self.bias,
            self.thresholds,
            self.maxima,
            self.max_steps_tensor,
            self.crossed,
            self.last_scores,
            self.result,
            self.act_rows,
            self.ok,
            self.nonfinite,
            D=width,
            P=count,
            BLOCK_D=int(os.environ.get("LMSM_ANYTIME_BLOCK_D", "512")),
            BLOCK_P=max(16, 1 << (count - 1).bit_length()),
            ACTIONS=self.actions_enabled,
            num_warps=int(
                os.environ.get("LMSM_ANYTIME_WARPS", "8" if self.max_rows <= 64 else "4")
            ),
        )
        self.pinned_ok.copy_(self.ok, non_blocking=True)
        if self.actions_enabled:
            self.pinned_result.copy_(self.result, non_blocking=True)

    def _step_functions(self):
        return {"step": self._step}

    def run(self, flags):
        del flags
        self._run("step", self._step)
        return self.actions_enabled

    def _archive_policy(self, index):
        counts = self.counts.index_select(0, index).cpu().numpy()
        maxima = self.maxima.index_select(0, index).cpu().numpy()
        steps = self.max_steps_tensor.index_select(0, index).cpu().numpy()
        crossed = self.crossed.index_select(0, index).cpu().numpy()
        records = []
        for position in range(index.shape[0]):
            evaluated = counts[position] > 0
            records.append(
                {
                    "policy_scores": (
                        {t: float(maxima[position, c]) for c, t in enumerate(self.target_ids)}
                        if evaluated
                        else {}
                    ),
                    "policy_score_steps": (
                        {t: int(steps[position, c]) for c, t in enumerate(self.target_ids)}
                        if evaluated
                        else {}
                    ),
                    "policy_actions": (
                        {t: bool(crossed[position, c]) for c, t in enumerate(self.target_ids)}
                        if evaluated
                        else {}
                    ),
                    "evaluation_error": None,
                }
            )
        return records

    def threshold_for(self, rule_index: int) -> float:
        return float(self.thresholds[rule_index].item())


class DeviceCheckpointEngine(_DeviceEngine):
    """GPU form of ``CheckpointPolicyEvaluator`` (prompt, two windows, fusion at 64)."""

    observes_prompt = True

    def __init__(self, evaluator: CheckpointPolicyEvaluator, num_slots, max_rows, max_steps, device):
        super().__init__(evaluator, num_slots, max_rows, max_steps, device)
        rules = list(evaluator.rules)
        self.target_ids = [rule.target_id for rule in rules]
        self.split = int(evaluator.window_split_step)
        self.evaluation_step = int(evaluator.evaluation_step)
        self.max_decode_step = int(evaluator.max_decode_step)
        self.prompt_weight, self.prompt_bias = _probe_matrices([r.prompt for r in rules], device)
        self.window_1_weight, self.window_1_bias = _probe_matrices(
            [r.steps_1_32 for r in rules], device
        )
        self.window_2_weight, self.window_2_bias = _probe_matrices(
            [r.steps_33_64 for r in rules], device
        )

        def stack(name: str) -> torch.Tensor:
            return torch.tensor(
                np.stack([np.asarray(getattr(r.fusion, name), dtype=np.float64) for r in rules]),
                dtype=torch.float64,
                device=device,
            )

        self.fusion_mean = stack("mean")
        self.fusion_scale = stack("scale")
        self.fusion_coef = stack("coef")
        self.fusion_intercept = torch.tensor(
            [float(r.fusion.intercept) for r in rules], dtype=torch.float64, device=device
        )
        self.fusion_threshold = torch.tensor(
            [float(r.fusion.threshold) for r in rules], dtype=torch.float64, device=device
        )
        width = self.prompt_weight.shape[0]
        self.width = int(width)
        count = len(rules)
        slots = self.num_slots + 1
        self.window_sums = torch.zeros((slots * 2, width), dtype=torch.float32, device=device)
        self.window_counts = torch.zeros((slots * 2,), dtype=torch.float32, device=device)
        self.scores = torch.zeros((slots, 4, count), dtype=torch.float64, device=device)
        # done[:, 0..3]: prompt, window 1, window 2, fusion scored.
        self.done = torch.zeros((slots, 4), dtype=torch.bool, device=device)
        self.crossed = torch.zeros((slots, count), dtype=torch.bool, device=device)
        self.errors = torch.zeros((slots, 2), dtype=torch.bool, device=device)

    def _reset_policy_slots(self, slots):
        window_slots = torch.cat([slots * 2, slots * 2 + 1])
        self.window_sums.index_fill_(0, window_slots, 0.0)
        self.window_counts.index_fill_(0, window_slots, 0.0)
        self.scores.index_fill_(0, slots, 0.0)
        self.done.index_fill_(0, slots, False)
        self.crossed.index_fill_(0, slots, False)
        self.errors.index_fill_(0, slots, False)

    def _policy_slot_is_fresh(self, index):
        window = torch.cat([index * 2, index * 2 + 1])
        return bool(
            (self.window_sums.index_select(0, window) == 0).all().item()
            and (self.window_counts.index_select(0, window) == 0).all().item()
            and not self.done.index_select(0, index).any().item()
            and not self.crossed.index_select(0, index).any().item()
            and not self.errors.index_select(0, index).any().item()
        )

    def _accumulate(self) -> None:
        if self.use_triton:
            from lmsm.integrations.vllm_device_kernels import checkpoint_accumulate_kernel

            self._instrument_rows()
            checkpoint_accumulate_kernel[(self.max_rows,)](
                self.index,
                self.max_rows,
                self.captured_epoch,
                self.capture_buffer,
                self.capture_buffer.stride(0),
                self.observed,
                self.observed.stride(0),
                self.max_steps,
                self.window_sums,
                self.window_counts,
                self.ok,
                self.nonfinite,
                self.split,
                self.max_decode_step,
                D=self.window_sums.shape[1],
                BLOCK_D=256,
            )
            self.pinned_ok.copy_(self.ok, non_blocking=True)
            return
        hidden = self._prologue()
        _rows, slots, steps, _tokens, valid = self._inputs()
        self._record_nonfinite(slots, steps, valid & ~torch.isfinite(hidden).all(dim=1))
        accumulate = (steps >= 1) & (steps <= self.max_decode_step) & valid
        flat = slots * 2 + (steps > self.split).to(torch.int64)
        weight = accumulate.to(torch.float32)
        self.window_sums.index_add_(0, flat, hidden * weight.unsqueeze(1))
        self.window_counts.index_add_(0, flat, weight)

    def _set_phase(self, slots, select, phase: int, values: torch.Tensor) -> None:
        scores = self.scores.index_select(0, slots)
        scores[:, phase, :] = torch.where(select.unsqueeze(1), values, scores[:, phase, :])
        self.scores.index_copy_(0, slots, scores)
        done = self.done.index_select(0, slots)
        done[:, phase] |= select
        self.done.index_copy_(0, slots, done)

    def _prompt(self) -> None:
        _rows, slots, steps, tokens, valid = self._inputs()
        select = (steps == 0) & valid
        hidden = self.capture_buffer.index_select(0, tokens).to(torch.float32)
        prompt = torch.addmm(self.prompt_bias, hidden.to(torch.float64), self.prompt_weight)
        self._record_nonfinite(slots, steps, select & ~torch.isfinite(prompt).all(dim=1))
        self._set_phase(slots, select, 0, prompt)

    def _window_1(self) -> None:
        _rows, slots, steps, _tokens, valid = self._inputs()
        select = (steps == self.split) & valid
        count = self.window_counts.index_select(0, slots * 2)
        pooled = self.window_sums.index_select(0, slots * 2) / count.clamp_min(1.0).unsqueeze(1)
        window_1 = torch.addmm(self.window_1_bias, pooled.to(torch.float64), self.window_1_weight)
        self._record_nonfinite(
            slots,
            steps,
            select & ~(torch.isfinite(pooled).all(dim=1) & torch.isfinite(window_1).all(dim=1)),
        )
        self._set_phase(slots, select, 1, window_1)
        errors = self.errors.index_select(0, slots)
        errors[:, 0] |= select & (count != float(self.split))
        self.errors.index_copy_(0, slots, errors)

    def _evaluate(self) -> None:
        rows, slots, steps, _tokens, valid = self._inputs()
        select = (steps == self.evaluation_step) & valid
        count = self.window_counts.index_select(0, slots * 2 + 1)
        pooled = self.window_sums.index_select(0, slots * 2 + 1) / count.clamp_min(1.0).unsqueeze(
            1
        )
        window_2 = torch.addmm(self.window_2_bias, pooled.to(torch.float64), self.window_2_weight)
        scores = self.scores.index_select(0, slots)
        components = torch.stack([scores[:, 0, :], scores[:, 1, :], window_2], dim=-1)
        fusion = (
            ((components - self.fusion_mean) / self.fusion_scale) * self.fusion_coef
        ).sum(dim=-1) + self.fusion_intercept
        finite = (
            torch.isfinite(pooled).all(dim=1)
            & torch.isfinite(window_2).all(dim=1)
            & torch.isfinite(fusion).all(dim=1)
        )
        failed = self._record_nonfinite(slots, steps, select & ~finite)
        scores[:, 2, :] = torch.where(select.unsqueeze(1), window_2, scores[:, 2, :])
        scores[:, 3, :] = torch.where(select.unsqueeze(1), fusion, scores[:, 3, :])
        self.scores.index_copy_(0, slots, scores)
        done = self.done.index_select(0, slots)
        complete = done[:, 0] & done[:, 1]
        done[:, 2] |= select
        done[:, 3] |= select
        self.done.index_copy_(0, slots, done)
        expected = float(self.evaluation_step - self.split)
        errors = self.errors.index_select(0, slots)
        errors[:, 1] |= select & ((count != expected) | ~complete)
        self.errors.index_copy_(0, slots, errors)
        crossed = (fusion >= self.fusion_threshold) & select.unsqueeze(1)
        current = self.crossed.index_select(0, slots)
        self.crossed.index_copy_(0, slots, torch.where(select.unsqueeze(1), crossed, current))
        if not self.actions_enabled:
            return
        triggered = crossed.any(dim=1) & ~failed
        rule = torch.argmax(crossed.to(torch.int32), dim=1)
        picked = rule.unsqueeze(1)
        self.result[:, 0] = triggered.to(torch.float64)
        self.result[:, 1] = rule.to(torch.float64)
        self.result[:, 2] = fusion.gather(1, picked).squeeze(1)
        self.result[:, 3:6] = components.gather(1, picked.unsqueeze(2).expand(-1, 1, 3)).squeeze(1)
        self.act_rows.fill_(False)
        self.act_rows.index_copy_(0, rows, triggered)
        self.pinned_result.copy_(self.result, non_blocking=True)

    def _step_functions(self):
        return {
            "accumulate": self._accumulate,
            "prompt": self._prompt,
            "window_1": self._window_1,
            "evaluate": self._evaluate,
        }

    def run(self, flags):
        self._run("accumulate", self._accumulate)
        if flags.any_prompt:
            self._run("prompt", self._prompt)
        if flags.any_window_1_end:
            self._run("window_1", self._window_1)
        if flags.any_evaluation:
            self._run("evaluate", self._evaluate)
            return self.actions_enabled
        return False

    def _archive_policy(self, index):
        scores = self.scores.index_select(0, index).cpu().numpy()
        done = self.done.index_select(0, index).cpu().numpy()
        crossed = self.crossed.index_select(0, index).cpu().numpy()
        errors = self.errors.index_select(0, index).cpu().numpy()
        names = ("prompt", "steps_1_32", "steps_33_64", "fusion")
        records = []
        for position in range(index.shape[0]):
            policy_scores = {
                name: (
                    {t: float(scores[position, phase, c]) for c, t in enumerate(self.target_ids)}
                    if done[position, phase]
                    else {}
                )
                for phase, name in enumerate(names)
            }
            error = None
            if errors[position, 0]:
                error = f"LMSM-Checkpoint requires generated hidden states 1 through {self.split}"
            if errors[position, 1]:
                error = (
                    "LMSM-Checkpoint requires the prompt, window 1, and generated hidden "
                    f"states {self.split + 1} through {self.evaluation_step}"
                )
            records.append(
                {
                    "policy_scores": policy_scores,
                    "policy_score_steps": {
                        t: self.evaluation_step
                        for c, t in enumerate(self.target_ids)
                        if crossed[position, c]
                    },
                    "policy_actions": {
                        t: bool(crossed[position, c]) for c, t in enumerate(self.target_ids)
                    },
                    "evaluation_error": error,
                    # The evaluation point counts as evaluated only if the fusion
                    # phase itself ran for this slot, not merely because the
                    # step's observation was delivered.
                    "evaluated_points": [self.evaluation_step] if done[position, 3] else [],
                }
            )
        return records

    def threshold_for(self, rule_index: int) -> float:
        return float(self.fusion_threshold[rule_index].item())


class UnsupportedDeviceBackend(ValueError):
    """The backend cannot be evaluated exactly on the device path."""

    requirement = "elementwise_selected_feature_encoder"


class DuplicateFeatureIds(UnsupportedDeviceBackend):
    """A selected-feature rule lists a feature ID more than once."""

    requirement = "distinct_selected_feature_ids"


def _check_distinct_feature_ids(union: Any, backends: list[Any]) -> None:
    """Refuse duplicate feature IDs, which the device engine cannot reproduce.

    The engine keeps one column per selected feature and one threshold per
    rule and column.  The host path counts every entry of a rule's feature
    list, each with its own threshold.  A rule that lists a feature twice would
    collapse into one column on the device, while the host keeps both entries
    in its pass fraction.  A feature shared by different rules is not a
    duplicate, because each rule has its own threshold row.
    """

    def repeated_ids(features: Any) -> list[int]:
        counts = Counter(int(feature) for feature in features)
        return sorted(feature for feature, count in counts.items() if count > 1)

    requirement = DuplicateFeatureIds.requirement
    repeated = repeated_ids(union)
    if repeated:
        raise DuplicateFeatureIds(
            f"{requirement}: the gathered feature selection lists feature(s) {repeated} "
            "more than once"
        )
    for backend in backends:
        repeated = repeated_ids(backend.feature_indices)
        if repeated:
            raise DuplicateFeatureIds(
                f"{requirement}: rule {backend.name!r} lists feature(s) {repeated} more "
                "than once; the host path counts each entry with its own threshold, the "
                "device path keeps one column per feature, so this policy runs with "
                "host evaluation"
            )


class SelectedEncoder:
    """The referenced features of a feature dictionary, with the host's operations.

    ``relu_linear`` (``LinearTranscoderAdapter``): ``relu(x @ W_sel.T + b_sel)``
    in the artifact dtype, operation for operation the host path's
    ``encode_selected``.

    ``jumprelu_sae`` (sae_lens ``JumpReLUSAE``): sae_lens ``encode`` restricted
    to the referenced features.  ``process_sae_in`` casts to the SAE dtype,
    applies no reshape, no hook and no run-time normalization (both are
    refused otherwise), and subtracts ``b_dec`` only when
    ``apply_b_dec_to_input``.  Then ``pre = x @ W_enc[:, sel] + b_enc[sel]``
    and the output is ``relu(pre) * (pre > threshold[sel])``.  JumpReLU is
    elementwise, so the selected features are exactly the full encode's
    selected columns.  Only the matmul's reduction order can differ from the
    host path's full-width matmul, because different GEMM shapes may use
    different kernels.
    """

    def __init__(self, kind, weight, bias, *, input_dtype=None, input_bias=None, jump_threshold=None):
        self.kind = kind
        self.weight = weight
        self.bias = bias
        self.input_dtype = input_dtype
        self.input_bias = input_bias
        self.jump_threshold = jump_threshold

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.kind == "relu_linear":
            return torch.relu(hidden @ self.weight.transpose(0, 1) + self.bias)
        x = hidden.to(self.input_dtype)
        if self.input_bias is not None:
            x = x - self.input_bias
        pre = x @ self.weight + self.bias
        return torch.relu(pre) * (pre > self.jump_threshold).to(pre.dtype)


def _refuse(detail: str) -> UnsupportedDeviceBackend:
    return UnsupportedDeviceBackend(f"{UnsupportedDeviceBackend.requirement}: {detail}")


def _check_selected_encoder(artifact: Any) -> str:
    """The encoder kind the device path evaluates exactly, or raise."""

    from lmsm.loaders.sae_loader import LinearTranscoderAdapter

    if isinstance(artifact, LinearTranscoderAdapter):
        return "relu_linear"
    cls = type(artifact)
    if cls.__name__ == "JumpReLUSAE" and cls.__module__.startswith("sae_lens."):
        cfg = artifact.cfg
        if getattr(cfg, "normalize_activations", "none") != "none":
            raise _refuse(
                f"sae_lens run-time normalization {cfg.normalize_activations!r} is not "
                "elementwise per feature; this SAE runs on the host evaluation path"
            )
        if getattr(cfg, "reshape_activations", "none") != "none":
            raise _refuse(f"sae_lens input reshape {cfg.reshape_activations!r} is not supported")
        if not isinstance(artifact.activation_fn, torch.nn.ReLU):
            raise _refuse(f"JumpReLU base activation {type(artifact.activation_fn).__name__}")
        for name in ("hook_sae_input", "hook_sae_acts_pre", "hook_sae_acts_post"):
            hook = getattr(artifact, name)
            if hook._forward_hooks or hook._forward_pre_hooks:
                raise _refuse(f"sae_lens hook {name} has hooks attached")
        return "jumprelu_sae"
    raise _refuse(
        "device evaluation supports selected-feature artifacts with an elementwise "
        "encoder (LinearTranscoderAdapter, sae_lens JumpReLUSAE); "
        f"{cls.__name__} runs on the host evaluation path"
    )


def _selected_encoder(
    artifact: Any, feature_indices: list[int], device: torch.device | None = None
) -> SelectedEncoder:
    """Gather the referenced features' encoder parameters once, at startup.

    The gathered parameters are moved to ``device``, the engine's GPU, so the
    artifact itself may stay on the host.  With vLLM's background engine core
    the evaluator is built on the CPU.  Initializing CUDA in the parent would
    make vLLM spawn the engine process instead of forking it, and a spawned
    process lacks the configured integration.  Artifacts whose selected
    encoding is not exact (non-elementwise activations such as TopK, run-time
    normalization, unknown SAE classes) are refused with a named requirement.
    """

    kind = _check_selected_encoder(artifact)
    target = device if device is not None else artifact.W_enc.device
    index = torch.tensor(feature_indices, dtype=torch.long, device=artifact.W_enc.device)
    with torch.no_grad():
        if kind == "relu_linear":
            return SelectedEncoder(
                kind,
                artifact.W_enc.index_select(0, index).contiguous().to(target),
                artifact.b_enc.index_select(0, index).contiguous().to(target),
            )
        input_bias = (
            artifact.b_dec.detach().clone().to(target) if artifact.cfg.apply_b_dec_to_input else None
        )
        return SelectedEncoder(
            kind,
            artifact.W_enc.detach().index_select(1, index).contiguous().to(target),
            artifact.b_enc.detach().index_select(0, index).contiguous().to(target),
            input_dtype=artifact.dtype,
            input_bias=input_bias,
            jump_threshold=artifact.threshold.detach().index_select(0, index).contiguous().to(target),
        )


class DeviceSharedFeatureEngine(_DeviceEngine):
    """GPU form of ``SharedFeaturePolicyEvaluator`` for selected-feature backends.

    At each step, the selected encoder parameters, gathered once at startup,
    give the referenced feature activations with the host path's operations
    (``SelectedEncoder``).  The rest follows the host backends, aggregator,
    and decider.  This covers per-feature running maxima, per-rule step and
    cumulative pass fractions, the max-score aggregation per target, threshold
    crossings, and the first triggered rule in active-rule order.  The encode
    runs on exactly the eligible rows, without padding.  For the transcoder it
    is the host's matmul with the same shape, dtype and bias.  For a JumpReLU
    SAE the host encodes the full dictionary, so only the matmul's reduction
    order can differ.  A rule that lists a feature ID twice is refused
    (``distinct_selected_feature_ids``).

    Every step also checks the eligible rows' evidence for NaN and infinite
    values (``_check_evidence``).  On CUDA this is one fused Triton kernel next
    to the PyTorch arithmetic, selected by ``use_triton``.
    """

    observes_prompt = True
    observes_decode = True
    max_decode_step = None

    def __init__(self, evaluator: SharedFeaturePolicyEvaluator, num_slots, max_rows, max_steps, device):
        super().__init__(evaluator, num_slots, max_rows, max_steps, device)
        self.use_graphs = False
        # Only the finiteness check has a Triton kernel, and it needs CUDA.
        self.use_triton = self.use_triton and torch.device(device).type == "cuda"
        shared = evaluator.shared_backend
        _check_distinct_feature_ids(shared.feature_indices, list(evaluator.backend_templates))
        self.union = list(shared.feature_indices)
        position = {feature: column for column, feature in enumerate(self.union)}
        self.encoder = _selected_encoder(shared.artifact, self.union, device)
        backends = list(evaluator.backend_templates)
        profile = evaluator.profile
        self.backend_categories = [backend.category for backend in backends]
        self.backend_features = [list(backend.feature_indices) for backend in backends]
        count = len(backends)
        width = len(self.union)
        thresholds = np.full((count, width), np.inf, dtype=np.float64)
        sizes = np.zeros(count, dtype=np.float64)
        cumulative = np.zeros(count, dtype=bool)
        for row, backend in enumerate(backends):
            if backend.feature_group_rule not in ("any", "fraction", "cumulative_fraction"):
                raise UnsupportedDeviceBackend(
                    f"{UnsupportedDeviceBackend.requirement}: unknown feature group rule "
                    f"{backend.feature_group_rule!r}"
                )
            for feature, threshold in zip(backend.feature_indices, backend.feature_thresholds):
                thresholds[row, position[feature]] = float(threshold)
            sizes[row] = max(len(backend.feature_indices), 1)
            cumulative[row] = backend.feature_group_rule == "cumulative_fraction"
        self.thresholds = torch.tensor(thresholds, dtype=torch.float64, device=device)
        self.sizes = torch.tensor(sizes, dtype=torch.float64, device=device)
        self.cumulative = torch.tensor(cumulative, dtype=torch.bool, device=device)
        # MaxScoreAggregator gives, per active rule, the mean score of the signals
        # whose category is the rule's target.  ThresholdPolicy picks the first
        # triggered rule in active-rule order.
        rules = list(profile.active_rules)
        self.target_ids = [rule.target_id for rule in rules]
        incidence = np.zeros((count, len(rules)), dtype=np.float64)
        for column, rule in enumerate(rules):
            for row, category in enumerate(self.backend_categories):
                if category == rule.target_id:
                    incidence[row, column] = 1.0
        members = incidence.sum(axis=0)
        self.incidence = torch.tensor(incidence, dtype=torch.float64, device=device)
        self.members = torch.tensor(members, dtype=torch.float64, device=device)
        self.rule_thresholds = torch.tensor(
            [float(rule.condition.config.get("threshold", 1.0)) for rule in rules],
            dtype=torch.float64,
            device=device,
        )
        self.actions_enabled = bool(getattr(evaluator.decider, "actions_enabled", True))
        slots = self.num_slots + 1
        self.maxima = torch.zeros((slots, width), dtype=torch.float64, device=device)
        self.evaluated = torch.zeros((slots,), dtype=torch.bool, device=device)
        # Per eligible row: the slot has no non-finite record (kernel output).
        self.allowed = torch.ones((self.max_rows,), dtype=torch.bool, device=device)
        self.loaded_rows = 0

    def _reset_policy_slots(self, slots):
        self.maxima.index_fill_(0, slots, 0.0)
        self.evaluated.index_fill_(0, slots, False)

    def _policy_slot_is_fresh(self, index):
        return bool(
            (self.maxima.index_select(0, index) == 0).all().item()
            and not self.evaluated.index_select(0, index).any().item()
        )

    def _step_functions(self):
        return {}

    def capture_graphs(self) -> None:
        return None

    def run(self, flags):
        del flags
        assert self.capture_buffer is not None and self.captured_epoch is not None
        m = self.loaded_rows
        index = self.index
        self.ok.copy_(self.captured_epoch == index[EPOCH, 0:1])
        rows, slots, steps, tokens = (
            index[ROW, :m], index[SLOT, :m], index[STEP, :m], index[TOKEN, :m]
        )
        ok = self.ok
        hidden = self.capture_buffer.index_select(0, tokens)
        if self.instrument:
            self.last_hidden[:m].copy_(hidden)
        self.observed.index_put_(
            (slots, steps.clamp(0, self.max_steps - 1)), ok.to(torch.uint8).expand(m)
        )
        # The host encode restricted to the referenced features, on the m eligible rows.
        scores = self.encoder.encode(hidden).to(torch.float64)
        allowed = self._check_evidence(slots, steps, hidden, scores)
        previous = self.maxima.index_select(0, slots)
        maxima = torch.where(ok.unsqueeze(1), torch.maximum(previous, scores), previous)
        self.maxima.index_copy_(0, slots, maxima)
        self.evaluated.index_copy_(0, slots, self.evaluated.index_select(0, slots) | ok)
        step_passes = (scores.unsqueeze(1) >= self.thresholds).sum(dim=2).to(torch.float64)
        cumulative_passes = (maxima.unsqueeze(1) >= self.thresholds).sum(dim=2).to(torch.float64)
        signal = torch.where(
            self.cumulative, cumulative_passes / self.sizes, step_passes / self.sizes
        )
        risk = (signal @ self.incidence) / self.members.clamp_min(1.0)
        risk = torch.where(self.members > 0, risk, torch.zeros_like(risk))
        triggered_rules = (risk >= self.rule_thresholds) & ok.unsqueeze(1)
        self.pinned_ok.copy_(self.ok, non_blocking=True)
        if not self.actions_enabled:
            return False
        triggered = triggered_rules.any(dim=1) & allowed
        rule = torch.argmax(triggered_rules.to(torch.int32), dim=1)
        self.result[:m, 0] = triggered.to(torch.float64)
        self.result[:m, 1] = rule.to(torch.float64)
        self.result[:m, 2] = risk.gather(1, rule.unsqueeze(1)).squeeze(1)
        self.act_rows.fill_(False)
        self.act_rows.index_copy_(0, rows, triggered)
        self.pinned_result[:m].copy_(self.result[:m], non_blocking=True)
        return True

    def _check_evidence(
        self, slots: torch.Tensor, steps: torch.Tensor, hidden: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor | None:
        """Record each row's first non-finite step and return which rows' slots may act.

        For rows whose capture was delivered (``ok``), this checks every
        gathered activation value and every selected feature activation.  The
        gathered values matter because a -inf activation can leave the features
        finite, since relu(-inf) = 0.  A slot with a record (this step or
        earlier) takes no action.  On CUDA one fused kernel does all of it in
        one launch with no host synchronization.  The PyTorch form below is the
        reference (CPU, ``LMSM_DEVICE_TRITON=0``).  Returns None when actions
        are disabled.
        """

        if self.use_triton:
            from lmsm.integrations.vllm_device_kernels import selected_feature_nonfinite_kernel

            m, width = hidden.shape
            count = int(scores.shape[1])
            assert hidden.stride(1) == 1 and scores.stride(1) == 1
            selected_feature_nonfinite_kernel[(int(m),)](
                self.index,
                self.max_rows,
                self.ok,
                hidden,
                hidden.stride(0),
                scores,
                scores.stride(0),
                self.nonfinite,
                self.allowed,
                D=int(width),
                F=count,
                BLOCK_D=min(1024, max(16, 1 << (int(width) - 1).bit_length())),
                BLOCK_F=min(1024, max(16, 1 << (count - 1).bit_length())),
                ACTIONS=self.actions_enabled,
                num_warps=4,
            )
            return self.allowed[: int(m)] if self.actions_enabled else None
        bad = self.ok & ~(torch.isfinite(hidden).all(dim=1) & torch.isfinite(scores).all(dim=1))
        failed = self._record_nonfinite(slots, steps, bad)
        return ~failed if self.actions_enabled else None

    def _archive_policy(self, index):
        maxima = self.maxima.index_select(0, index).cpu().numpy()
        evaluated = self.evaluated.index_select(0, index).cpu().numpy()
        column = {feature: position for position, feature in enumerate(self.union)}
        records = []
        for position in range(index.shape[0]):
            feature_maxima: dict[str, dict[int, float]] = {}
            if evaluated[position]:
                for category, features in zip(
                    self.backend_categories, self.backend_features, strict=True
                ):
                    target = feature_maxima.setdefault(category, {})
                    for feature in features:
                        target[feature] = float(maxima[position, column[feature]])
            records.append(
                {
                    "policy_scores": {},
                    "policy_score_steps": {},
                    "policy_actions": {},
                    "feature_maxima": feature_maxima,
                    "evaluation_error": None,
                }
            )
        return records

    def threshold_for(self, rule_index: int) -> float:
        return float(self.rule_thresholds[rule_index].item())


def check_device_evaluator(evaluator: Any) -> None:
    """Raise ``UnsupportedDeviceBackend`` unless device evaluation is exact here.

    Called before the engine is constructed, so a deployment is refused at
    startup with the named requirement instead of failing inside vLLM.
    """

    if isinstance(evaluator, (AnytimePolicyEvaluator, CheckpointPolicyEvaluator)):
        return
    if isinstance(evaluator, SharedFeaturePolicyEvaluator):
        _check_selected_encoder(evaluator.shared_backend.artifact)
        for backend in evaluator.backend_templates:
            if backend.feature_group_rule not in ("any", "fraction", "cumulative_fraction"):
                raise UnsupportedDeviceBackend(
                    f"{UnsupportedDeviceBackend.requirement}: unknown feature group rule "
                    f"{backend.feature_group_rule!r}"
                )
        _check_distinct_feature_ids(
            evaluator.shared_backend.feature_indices, list(evaluator.backend_templates)
        )
        return
    raise UnsupportedDeviceBackend(
        f"{UnsupportedDeviceBackend.requirement}: no device engine for "
        f"{type(evaluator).__name__}"
    )


def build_device_engine(
    evaluator: Any, num_slots: int, max_rows: int, max_steps: int, device: torch.device
) -> _DeviceEngine:
    if isinstance(evaluator, AnytimePolicyEvaluator):
        return DeviceAnytimeEngine(evaluator, num_slots, max_rows, max_steps, device)
    if isinstance(evaluator, CheckpointPolicyEvaluator):
        return DeviceCheckpointEngine(evaluator, num_slots, max_rows, max_steps, device)
    if isinstance(evaluator, SharedFeaturePolicyEvaluator):
        return DeviceSharedFeatureEngine(evaluator, num_slots, max_rows, max_steps, device)
    raise ValueError(
        "device evaluation supports the dense-probe Anytime and Checkpoint "
        f"evaluators, not {type(evaluator).__name__}"
    )


__all__ = [
    "DeviceAnytimeEngine",
    "DeviceCheckpointEngine",
    "DeviceSharedFeatureEngine",
    "DuplicateFeatureIds",
    "SelectedEncoder",
    "UnsupportedDeviceBackend",
    "check_device_evaluator",
    "StepFlags",
    "build_device_engine",
]
