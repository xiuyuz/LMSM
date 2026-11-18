"""CUDA robustness checks of the device engines and kernels (no vLLM engine, no model).

Run on one GPU: ``python -m pytest -q tests/test_device_robustness_cuda.py``.
Without CUDA every test is skipped, so the CPU suite reports them as skipped.

The checks drive the device engines and the device-mode runtime directly.
They use synthetic policies and a simulated capture op, in which each forward
writes its packed rows to the capture buffer and stamps the epoch.  They run
the default execution path, with step functions captured as CUDA graphs and
the fused Triton kernels compiled for this GPU.  Where it is informative, they
compare the result with the PyTorch reference on the same GPU.  The CPU tests
run the same cases only on the CPU or under Triton's interpreter.

* Hidden widths that the kernel blocks do not divide (bounds masks).
* NaN, +inf and -inf in one request's activation rows and scores.  Only that
  request fails closed, every other request is bitwise identical to a clean
  run, and the kernels agree with the reference.
* Finite activations whose running sums or feature activations overflow.
* A Checkpoint request whose fusion evaluation was skipped (fails closed).
* Mixed clean and invalid requests in one batch with departures, admissions
  and row moves.
* A slot reused after a poisoned request (no sticky failure).
* A non-finite value only on a discarded asynchronous forward (completes).

The scenario helpers live in ``test_graph_observation.py`` and take a device.
This file adds the batch flows that need real batch updates.
"""

from __future__ import annotations

import dataclasses
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


CUDA = "cuda"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
POLICIES = ["anytime", "checkpoint", "transcoder"]
NAN, INF = float("nan"), float("inf")


@pytest.fixture(scope="module")
def base():
    """The CPU test module's scenario helpers (loaded only when a test runs)."""

    return runpy.run_path(str(Path(__file__).with_name("test_graph_observation.py")))


@pytest.fixture(autouse=True)
def _isolate():
    saved = os.environ.get("LMSM_ANYTIME_BLOCK_D")
    yield
    if saved is None:
        os.environ.pop("LMSM_ANYTIME_BLOCK_D", None)
    else:
        os.environ["LMSM_ANYTIME_BLOCK_D"] = saved
    from lmsm.integrations import vllm as runtime
    from lmsm.integrations.vllm_graph import CAPTURE

    runtime.STATE.reset()
    CAPTURE.reset()


# ---------------------------------------------------------------------------
# Batch flows through the device-mode runtime


def _batch_update(batch, leaving, joining):
    """vLLM's input-batch bookkeeping for one update.  Mutates ``batch``.

    A departing request leaves a hole.  A joining request takes the lowest
    hole, or the end if there is none.  Condense then moves the last request
    into each remaining hole, lowest first.
    """

    removed = []
    for request_id in leaving:
        row = batch.index(request_id)
        batch[row] = None
        removed.append(row)
    holes = sorted(removed)
    added = []
    for request_id in joining:
        row = holes.pop(0) if holes else len(batch)
        if row == len(batch):
            batch.append(request_id)
        else:
            batch[row] = request_id
        added.append((row, request_id))
    moved = []
    for hole in holes:
        while batch and batch[-1] is None:
            batch.pop()
        if hole >= len(batch):
            continue
        moved.append((len(batch) - 1, hole))
        batch[hole] = batch.pop()
    while batch and batch[-1] is None:
        batch.pop()
    return removed, added, moved, len(batch)


def _flow(base, policy, plan, *, device, triton, graphs, async_=False, skip_evaluation_at=()):
    """Run ``plan`` through the device-mode runtime.

    ``plan`` maps a request ID to (admit_at, length, values): the request
    joins before forward ``admit_at``, finishes after ``length`` tokens
    (sooner if the policy acts), and ``values(step)`` is its activation at
    each processed step.  Under ``async_`` a finished request runs one more
    forward whose sampled token is discarded.  ``skip_evaluation_at`` lists
    forwards whose Checkpoint fusion phase is skipped.  Returns the release
    views, the counters, each request's slot, and whether that slot was fresh
    on admission.
    """

    from lmsm.integrations import vllm as runtime
    from lmsm.integrations.vllm_graph import CAPTURE

    device = torch.device(device)
    state = runtime.STATE
    state.reset()
    state.configure(
        base["_nonfinite_scenario"](policy, poisoned=False).evaluator,
        [91, 92],
        observation_mode="graph",
        evaluation_mode="device",
    )
    state.eos_token_id = 3
    CAPTURE.reset()
    CAPTURE.allocate(64, 2, torch.float32, device)
    CAPTURE.captured_epoch = torch.zeros((1,), dtype=torch.int64, device=device)
    rows = 16
    state.init_device(max_rows=rows, max_steps=128, device=device)
    state.async_scheduling = async_
    if device.type != "cuda":
        state.device_event = base["_Event"]()
    engine = state.device_engine
    engine.use_graphs, engine.use_triton = graphs, triton
    clock = {"forward": 0}
    run = engine.run

    def run_with_skips(flags):
        if clock["forward"] in skip_evaluation_at:
            flags = dataclasses.replace(flags, any_evaluation=False)
        return run(flags)

    engine.run = run_with_skips
    state.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_tokens_no_spec=np.zeros(rows, dtype=np.int64),
            num_prompt_tokens=np.zeros(rows, dtype=np.int64),
            num_computed_tokens_cpu=np.zeros(rows, dtype=np.int64),
        )
    )
    move = SimpleNamespace(name="UNIDIRECTIONAL")
    prompt = [11, 12]
    outs = {request_id: [] for request_id in plan}
    pending = sorted(plan, key=lambda request_id: plan[request_id][0])
    batch: list[str] = []
    slots, fresh = {}, {}
    leaving: list[str] = []
    finishing: set[str] = set()
    while True:
        joining = [r for r in pending if plan[r][0] <= clock["forward"]]
        pending = [r for r in pending if r not in joining]
        if leaving or joining:
            removed, added, moved, size = _batch_update(batch, leaving, joining)
            state.update_batch(
                base["_update"](
                    batch_size=size,
                    removed=removed,
                    added=[(row, base["_params"](r), prompt, outs[r]) for row, r in added],
                    moved=[(source, target, move) for source, target in moved],
                )
            )
            for _, request_id in added:
                slots[request_id] = state.slot_by_request[request_id]
                fresh[request_id] = engine.slot_is_fresh(slots[request_id])
            leaving = []
        if not batch:
            if not pending:
                break
            clock["forward"] = min(plan[r][0] for r in pending)
            continue
        counts = [1 if outs[r] else len(prompt) for r in batch]
        base["_sync_batch"](state, {row: (len(prompt), outs[r]) for row, r in enumerate(batch)})
        state.begin_forward(counts)
        CAPTURE.buffer.zero_()
        for end, request_id in zip(np.cumsum(counts) - 1, batch, strict=True):
            values = plan[request_id][2](len(outs[request_id]))
            CAPTURE.buffer[end] = torch.tensor(values, dtype=torch.float32)
        CAPTURE.captured_epoch.copy_(CAPTURE.step_epoch)
        state.after_forward()
        logits = state.apply_device(torch.zeros(len(batch), 8, device=device)).cpu()
        done_now = []
        for row, request_id in enumerate(batch):
            if request_id in finishing:
                continue  # vLLM discards the token sampled in this forward
            acted = bool(torch.isinf(logits[row]).any())
            done = acted or len(outs[request_id]) + 1 == plan[request_id][1]
            outs[request_id].append(3 if done else 5)
            if done:
                done_now.append(request_id)
        if async_:
            leaving = [r for r in batch if r in finishing]
            finishing = set(done_now)
        else:
            leaving = done_now
        clock["forward"] += 1
    views = {r: runtime.buffered_output(r, list(outs[r])) for r in plan}
    counters = runtime.runtime_counters()
    state.reset()
    CAPTURE.reset()
    return views, counters, slots, fresh


def _clean(policy, fire_at=None):
    """Clean activations.  The policy fires at ``fire_at`` (Checkpoint at 64)."""

    def values(step):
        if policy == "anytime":
            # Running prefix mean of [0.5, 0.5] then a spike: crosses 3.0 exactly at fire_at.
            if fire_at is None:
                return [0.1, 0.1]
            return [fire_at + 1.0] * 2 if step == fire_at else [0.5, 0.5]
        if policy == "checkpoint":
            if fire_at is None:
                return [0.1, 0.1]
            return [1.0, 1.0] if step == 0 else [0.5, 0.5]
        return [2.0, 0.0] if step == fire_at else [0.1, 0.1]

    return values


def _poisoned(policy, poison):
    """Non-firing clean activations with ``poison`` (step -> values) applied."""

    clean = _clean(policy)
    return lambda step: poison.get(step, clean(step))


def _state(view):
    record = view["execution_record"]
    return record["state"], record["decision"], record["decision_step"]


def _same_release(a, b, path="view"):
    """Equal release views, with float fields (scores) to 1e-12 and everything else exact.

    The Anytime kernel and cuBLAS sum a probe's dot product in different
    orders, so a recorded score may differ in its last bits between the
    kernel path and the reference path.  Decisions, states and tokens may not.
    """

    import math

    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), path
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) or (a != a and b != b), path
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            _same_release(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            _same_release(x, y, f"{path}[{index}]")
        return
    assert a == b, (path, a, b)


def _failure_steps(view):
    record = view["execution_record"]
    return [f["step"] for f in record["runtime_failures"] if "non-finite" in f["reason"]]


MIXED = {
    # policy: (firing requests, poisoned requests with their expected failure step)
    "anytime": (
        {"f1": (0, 12, 3), "f2": (2, 12, 6), "late": (14, 6, 2)},
        {
            "pN": (0, 10, {2: [NAN, 0.1]}, 2),
            "pI": (3, 11, {4: [INF, 0.1]}, 4),
            "pM": (6, 8, {1: [-INF, 0.0]}, 1),
            "pO": (4, 12, {3: [-3.0e38, 0.0], 4: [-3.0e38, 0.0]}, 4),  # float32 sum overflow
        },
    ),
    "checkpoint": (
        {"f1": (0, 70, 64), "f2": (2, 72, 64), "late": (9, 66, 64)},
        {
            "pN": (0, 70, {0: [NAN, 1.0]}, 0),
            "pI": (3, 68, {20: [INF, 0.5]}, 20),
            "pM": (6, 51, {50: [-INF, 0.5]}, 50),
            "pO": (4, 70, {10: [3.0e38, 0.0], 11: [3.0e38, 0.0]}, 32),  # window-1 mean
        },
    ),
    "transcoder": (
        {"f1": (0, 12, 3), "f2": (2, 12, 6), "late": (14, 6, 2)},
        {
            "pN": (0, 10, {2: [NAN, 0.1]}, 2),
            "pI": (3, 11, {4: [INF, 0.1]}, 4),
            "pM": (6, 8, {1: [-INF, 0.0]}, 1),  # relu(-inf) = 0: features stay finite
            "pO": (4, 12, {3: [3.0e38, 3.0e38]}, 3),  # the feature activation overflows
        },
    ),
}
CLEAN_NONFIRING = {"n1": (0, 9), "n2": (1, 15), "n3": (5, 7), "n4": (8, 10)}


def _mixed_plan(policy, poisoned):
    firing, bad = MIXED[policy]
    scale = 1 if policy != "checkpoint" else 7  # Checkpoint requests reach step 64
    plan = {
        r: (admit, length * scale, _clean(policy))
        for r, (admit, length) in CLEAN_NONFIRING.items()
    }
    for r, (admit, length, fire_at) in firing.items():
        plan[r] = (admit, length, _clean(policy, fire_at))
    for r, (admit, length, poison, _) in bad.items():
        plan[r] = (admit, length, _poisoned(policy, poison) if poisoned else _clean(policy))
    return plan


def _check_mixed_batch(base, policy, *, device, triton, graphs):
    """Only the poisoned requests fail, at their steps, and the rest release the same views."""

    clean, clean_counters, _, _ = _flow(
        base, policy, _mixed_plan(policy, False), device=device, triton=triton, graphs=graphs
    )
    views, counters, _, _ = _flow(
        base, policy, _mixed_plan(policy, True), device=device, triton=triton, graphs=graphs
    )
    firing, bad = MIXED[policy]
    case = (policy, triton, str(device))
    for request_id in list(CLEAN_NONFIRING) + list(firing):
        assert views[request_id] == clean[request_id], (case, request_id)
    for request_id, (_, _, fire_at) in firing.items():
        assert _state(views[request_id]) == ("completed", "refuse", fire_at), (case, request_id)
    for request_id, (_, _, _, step) in bad.items():
        view = views[request_id]
        assert view["execution_record"]["state"] == "execution_failed", (case, request_id)
        assert _failure_steps(view) == [step], (case, request_id)
        assert view["released_output_tokens"] == [91, 92], (case, request_id)
        assert view["action_record"] is None, (case, request_id)
        assert clean[request_id]["execution_record"]["state"] != "execution_failed", (case, request_id)
    assert (counters["nonfinite_failures"], clean_counters["nonfinite_failures"]) == (4, 0), case
    assert counters["graph_epoch_failures"] == 0, case
    return views


def _check_slot_reuse(base, policy, *, device, triton, graphs):
    """A slot freed by a poisoned request carries nothing to its next request.

    ``p`` (poisoned) takes slot 0 and leaves.  ``q`` joins in the same batch
    update and takes p's row and slot.  It must release exactly what it
    releases when p was clean, and when q runs alone.
    """

    long = 7 if policy == "checkpoint" else 1
    p_length = 6 * long
    q_fire = 64 if policy == "checkpoint" else 3
    poison = {2: [NAN, 0.1]} if policy != "checkpoint" else {10: [NAN, 0.5]}
    results = []
    for p_values in (_poisoned(policy, poison), _clean(policy)):
        plan = {
            "p": (0, p_length, p_values),
            "o": (0, 12 * long, _clean(policy)),
            "q": (p_length, 10 * long, _clean(policy, q_fire)),
        }
        results.append(_flow(base, policy, plan, device=device, triton=triton, graphs=graphs))
    alone = _flow(
        base, policy, {"q": (0, 10 * long, _clean(policy, q_fire))},
        device=device, triton=triton, graphs=graphs,
    )
    (poisoned, _, slots, fresh), (clean, _, clean_slots, _) = results
    case = (policy, triton, str(device))
    assert slots["q"] == slots["p"] == clean_slots["q"] == 0, (case, slots)
    assert fresh["q"], case
    assert poisoned["p"]["execution_record"]["state"] == "execution_failed", case
    assert _failure_steps(poisoned["p"]) == [min(poison)], case
    assert poisoned["q"] == clean["q"] == alone[0]["q"], case
    assert _state(poisoned["q"]) == ("completed", "refuse", q_fire), case
    assert poisoned["o"] == clean["o"], case


def _check_skipped_fusion(base, *, device, triton, graphs):
    """A Checkpoint request whose fusion phase did not run fails closed.

    ``x`` reaches step 64 at forward 64, ``y`` two forwards later.  Skipping
    the fusion phase at forward 64 leaves x's evaluation point unevaluated
    although its observation was delivered: x fails closed with the fixed
    refusal, y completes.  Without the skip, both complete.
    """

    fire = _clean("checkpoint", 64)
    plan = {"x": (0, 70, fire), "y": (2, 70, fire)}
    skipped, _, _, _ = _flow(
        base, "checkpoint", plan, device=device, triton=triton, graphs=graphs,
        skip_evaluation_at={64},
    )
    full, _, _, _ = _flow(base, "checkpoint", plan, device=device, triton=triton, graphs=graphs)
    record = skipped["x"]["execution_record"]
    assert record["state"] == "execution_failed"
    assert record["missing_evaluation_steps"] == [64] and record["missing_observation_steps"] == []
    assert skipped["x"]["released_output_tokens"] == [91, 92]
    assert _state(skipped["y"]) == ("completed", "refuse", 64)
    assert skipped["y"] == full["y"]
    assert _state(full["x"]) == ("completed", "refuse", 64)


# ---------------------------------------------------------------------------
# Tests (CUDA graphs and the fused kernels, the default path)


def test_cuda_defaults_use_graphs_and_the_fused_kernels(base):
    """The default device path on CUDA is the one these tests exercise."""

    if os.environ.get("LMSM_DEVICE_TRITON", "1") == "0" or os.environ.get("LMSM_DEVICE_GRAPHS", "1") == "0":
        pytest.skip("LMSM_DEVICE_TRITON or LMSM_DEVICE_GRAPHS disables the default path")
    from lmsm.integrations.vllm_device import build_device_engine

    for policy in POLICIES:
        evaluator = base["_nonfinite_scenario"](policy, poisoned=False).evaluator
        engine = build_device_engine(evaluator, 4, 4, 32, torch.device(CUDA))
        assert engine.use_triton, policy
        assert engine.use_graphs == (policy != "transcoder"), policy


KERNEL_BOUNDS_CASES = [
    ["anytime", 2560, 1024],  # Qwen3-4B width, 2.5 blocks
    ["anytime", 2560, None],  # the default block (512) divides it
    ["anytime", 40, 16],
    ["anytime", 40, None],  # the block is wider than the row
    ["checkpoint", 2600, None],  # 10 x 256 + 40
    ["checkpoint", 300, None],
]


def test_kernels_stay_inside_each_row_on_cuda(base):
    """Non-divisible widths: no read or write past a row, and the kernels equal the reference."""

    base["_check_kernel_bounds"](KERNEL_BOUNDS_CASES, device=CUDA)


def test_kernels_record_the_reference_nonfinite_steps_on_cuda(base):
    """NaN, +-inf, inf * 0 and float32 overflow: the kernels match the reference."""

    cases = base["DENSE_NONFINITE_CASES"] + [
        ["anytime", 2560, 1024, True],
        ["anytime", 2560, None, False],
        ["checkpoint", 2560, None, False],
    ]
    base["_check_dense_nonfinite_agreement"](cases, device=CUDA)


def test_selected_feature_kernel_matches_its_reference_on_cuda(base):
    """The fused selected-feature check: bounds, NaN, +-inf, overflow, no delivery, first step."""

    widths = ((40, 3), (1500, 70), (2560, 129))
    base["_check_feature_kernel"](device=CUDA, widths=widths)
    base["_check_feature_engine_kernel"](device=CUDA, widths=widths)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("policy", POLICIES)
def test_nonfinite_evidence_fails_only_that_request_on_cuda(base, policy):
    """NaN, +-inf and overflow in one request's rows: only it fails, and the kernels equal the reference."""

    kernel = base["_check_nonfinite_scenario"](policy, "device", triton=True, device=CUDA, graphs=True)
    reference = base["_check_nonfinite_scenario"](
        policy, "device", triton=False, device=CUDA, graphs=True
    )
    _same_release(kernel, reference)


@pytest.mark.parametrize("policy", POLICIES)
def test_mixed_batch_with_departures_and_admissions_on_cuda(base, policy):
    """Clean and poisoned requests share a batch that changes every few forwards."""

    kernel = _check_mixed_batch(base, policy, device=CUDA, triton=True, graphs=True)
    reference = _check_mixed_batch(base, policy, device=CUDA, triton=False, graphs=True)
    _same_release(kernel, reference)


@pytest.mark.parametrize("policy", POLICIES)
def test_slot_reused_after_a_poisoned_request_on_cuda(base, policy):
    _check_slot_reuse(base, policy, device=CUDA, triton=True, graphs=True)


def test_skipped_checkpoint_fusion_fails_closed_on_cuda(base):
    _check_skipped_fusion(base, device=CUDA, triton=True, graphs=True)


@pytest.mark.parametrize("policy", POLICIES)
def test_nonfinite_only_on_a_discarded_async_forward_completes_on_cuda(base, policy):
    """Async scheduling: a NaN only on the discarded forward does not fail the request."""

    kernel = base["_check_async_scenario"](policy, triton=True, device=CUDA, graphs=True)
    reference = base["_check_async_scenario"](policy, triton=False, device=CUDA, graphs=True)
    _same_release(kernel, reference)
