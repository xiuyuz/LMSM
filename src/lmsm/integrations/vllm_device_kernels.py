"""Fused Triton kernels for the per-step device engine (hot path only).

One program handles one padded row.  For the steps that run on every decode
step, each kernel reproduces the PyTorch reference in ``vllm_device``
operation for operation.

* Running sums are float32 and are updated with ``s + h * w``.  ``w`` is 0 or
  1, so the product is exact.  Division uses the correctly rounded ``div_rn``,
  so sums and pooled means are bitwise identical to the PyTorch and numpy
  paths.
* Probe scores are float64 dot products of the pooled mean with the folded
  probe weights.  Only the summation order differs from cuBLAS and numpy
  (1e-15 relative).

Rows are independent because every real row has its own slot.  Padding rows
all point at the dummy slot, whose contents are never read.

Every load and store along the hidden dimension is masked with
``offsets < D``.  A block size that does not divide ``D`` therefore never reads
or writes past a row, into another slot's evidence, another token's
activation, or past the end of a tensor.  When the block size divides ``D``,
as the defaults do for ``D = 2560``, every mask is all-true.

Each kernel also records, per slot, the first step whose evidence was NaN or
infinite.  For Checkpoint the evidence is the activation row.  For Anytime it
is the probe scores.  They are non-finite whenever a pooled mean value is,
because a NaN or infinite term makes every dot product NaN or infinite, even
with a zero weight.  Like the Checkpoint evaluation phase, the Anytime kernel
takes no action for such a slot.  These checks are comparisons next to the
arithmetic and do not change it.  A slot's entry is read only when the kernel
needs it and written only for a new record.  Finite evidence therefore adds no
memory traffic beyond the read that action suppression needs.

``selected_feature_nonfinite_kernel`` runs the same check for the
selected-feature engine, whose step is otherwise PyTorch.  One launch checks
every gathered activation value of each eligible row, which catches a -inf
that relu maps to 0, and every selected feature activation.  It records the
slot's first non-finite step.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _row_inputs(index_ptr, R, r):
    row = tl.load(index_ptr + 0 * R + r)
    slot = tl.load(index_ptr + 1 * R + r)
    step = tl.load(index_ptr + 2 * R + r)
    token = tl.load(index_ptr + 3 * R + r)
    valid = tl.load(index_ptr + 4 * R + r) > 0
    expected = tl.load(index_ptr + 5 * R)
    return row, slot, step, token, valid, expected


@triton.jit
def _not_finite(x):
    # False for every finite value, true for NaN and both infinities.
    return (tl.abs(x) < float("inf")) == 0


@triton.jit
def _record_nonfinite(nonfinite_ptr, slot, step, bad, read):
    """Keep the slot's first non-finite step: record ``step`` if ``bad`` and none yet.

    The slot's entry is read where ``read`` (which must cover ``bad``) and
    written only for a new record.  Returns whether the slot has a record
    (this step or earlier), valid where ``read``.
    """

    first = tl.load(nonfinite_ptr + slot, mask=read, other=-1)
    tl.store(nonfinite_ptr + slot, step, mask=bad & (first < 0))
    return bad | (first >= 0)


@triton.jit
def anytime_step_kernel(
    index_ptr,
    R,
    captured_epoch_ptr,
    buffer_ptr,
    buffer_stride,
    observed_ptr,
    observed_stride,
    max_steps,
    sums_ptr,
    counts_ptr,
    weight_ptr,
    bias_ptr,
    thresholds_ptr,
    maxima_ptr,
    max_step_ptr,
    crossed_ptr,
    last_scores_ptr,
    result_ptr,
    act_rows_ptr,
    ok_ptr,
    nonfinite_ptr,
    D: tl.constexpr,
    P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ACTIONS: tl.constexpr,
):
    r = tl.program_id(0)
    row, slot, step, token, valid, expected = _row_inputs(index_ptr, R, r)
    ok = tl.load(captured_epoch_ptr) == expected
    if r == 0:
        tl.store(ok_ptr, ok)
    live = valid & ok
    weight = live.to(tl.float32)
    clamped = tl.minimum(tl.maximum(step, 0), max_steps - 1)
    tl.store(observed_ptr + slot * observed_stride + clamped, live.to(tl.uint8))

    count = tl.load(counts_ptr + slot) + weight
    tl.store(counts_ptr + slot, count)
    denominator = tl.maximum(count, 1.0)

    probes = tl.arange(0, BLOCK_P)
    probe_mask = probes < P
    acc = tl.zeros([BLOCK_D, BLOCK_P], dtype=tl.float64)
    for start in range(0, D, BLOCK_D):
        offsets = start + tl.arange(0, BLOCK_D)
        in_row = offsets < D
        hidden = tl.load(
            buffer_ptr + token * buffer_stride + offsets, mask=in_row, other=0.0
        ).to(tl.float32)
        total = tl.load(sums_ptr + slot * D + offsets, mask=in_row, other=0.0) + hidden * weight
        tl.store(sums_ptr + slot * D + offsets, total, mask=in_row)
        pooled = tl.math.div_rn(total, denominator).to(tl.float64)
        # Masked lanes contribute 0 * 0: a weight read past row D would not.
        block = tl.load(
            weight_ptr + offsets[:, None] * P + probes[None, :],
            mask=in_row[:, None] & probe_mask[None, :],
            other=0.0,
        )
        acc += pooled[:, None] * block
    scores = tl.sum(acc, axis=0) + tl.load(bias_ptr + probes, mask=probe_mask, other=0.0)
    tl.store(last_scores_ptr + r * P + probes, scores, mask=probe_mask)
    # A NaN or infinite running sum (pooled mean) makes every score non-finite,
    # so the scores alone decide.  P >= 1, so there is at least one score.
    bad = live & (tl.max((_not_finite(scores) & probe_mask).to(tl.int32), axis=0) > 0)
    if ACTIONS:
        # Actions need the slot's earlier record too (only live rows can act).
        failed = _record_nonfinite(nonfinite_ptr, slot, step, bad, live)
    else:
        _record_nonfinite(nonfinite_ptr, slot, step, bad, bad)

    previous = tl.load(maxima_ptr + slot * P + probes, mask=probe_mask, other=float("-inf"))
    update = (scores > previous) & live
    maxima = tl.where(update, scores, previous)
    tl.store(maxima_ptr + slot * P + probes, maxima, mask=probe_mask)
    steps = tl.load(max_step_ptr + slot * P + probes, mask=probe_mask, other=-1)
    tl.store(max_step_ptr + slot * P + probes, tl.where(update, step, steps), mask=probe_mask)
    thresholds = tl.load(thresholds_ptr + probes, mask=probe_mask, other=float("inf"))
    crossed = (maxima >= thresholds) & probe_mask
    previous_crossed = tl.load(crossed_ptr + slot * P + probes, mask=probe_mask, other=0) != 0
    newly = crossed & (previous_crossed == 0)
    tl.store(crossed_ptr + slot * P + probes, crossed, mask=probe_mask)
    if ACTIONS:
        triggered = (tl.sum(newly.to(tl.int32), axis=0) > 0) & live & (failed == 0)
        first = tl.min(tl.where(newly, probes, BLOCK_P), axis=0)
        rule = tl.where(first < BLOCK_P, first, 0)
        score = tl.sum(tl.where(probes == rule, maxima, 0.0), axis=0)
        tl.store(result_ptr + r * 6 + 0, triggered.to(tl.float64))
        tl.store(result_ptr + r * 6 + 1, rule.to(tl.float64))
        tl.store(result_ptr + r * 6 + 2, score)
        tl.store(act_rows_ptr + row, triggered)


@triton.jit
def checkpoint_accumulate_kernel(
    index_ptr,
    R,
    captured_epoch_ptr,
    buffer_ptr,
    buffer_stride,
    observed_ptr,
    observed_stride,
    max_steps,
    window_sums_ptr,
    window_counts_ptr,
    ok_ptr,
    nonfinite_ptr,
    split,
    max_decode_step,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    row, slot, step, token, valid, expected = _row_inputs(index_ptr, R, r)
    ok = tl.load(captured_epoch_ptr) == expected
    if r == 0:
        tl.store(ok_ptr, ok)
    live = valid & ok
    clamped = tl.minimum(tl.maximum(step, 0), max_steps - 1)
    tl.store(observed_ptr + slot * observed_stride + clamped, live.to(tl.uint8))
    accumulate = (step >= 1) & (step <= max_decode_step) & live
    weight = accumulate.to(tl.float32)
    window = slot * 2 + (step > split).to(tl.int64)
    tl.store(window_counts_ptr + window, tl.load(window_counts_ptr + window) + weight)
    nonfinite = tl.zeros([BLOCK_D], dtype=tl.int32)
    for start in range(0, D, BLOCK_D):
        offsets = start + tl.arange(0, BLOCK_D)
        in_row = offsets < D
        # Masked lanes load 0.0, which is finite.
        hidden = tl.load(
            buffer_ptr + token * buffer_stride + offsets, mask=in_row, other=0.0
        ).to(tl.float32)
        nonfinite = nonfinite | _not_finite(hidden).to(tl.int32)
        total = (
            tl.load(window_sums_ptr + window * D + offsets, mask=in_row, other=0.0)
            + hidden * weight
        )
        tl.store(window_sums_ptr + window * D + offsets, total, mask=in_row)
    bad = live & (tl.max(nonfinite, axis=0) > 0)
    _record_nonfinite(nonfinite_ptr, slot, step, bad, bad)


@triton.jit
def selected_feature_nonfinite_kernel(
    index_ptr,
    R,
    ok_ptr,
    hidden_ptr,
    hidden_stride,
    scores_ptr,
    scores_stride,
    nonfinite_ptr,
    allowed_ptr,
    D: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_F: tl.constexpr,
    ACTIONS: tl.constexpr,
):
    """One eligible row of the selected-feature engine (no padding rows).

    Checks every gathered activation value of the row, which catches a -inf
    that relu maps to 0, and every selected feature activation.  When the
    step's capture was delivered and a value is NaN or infinite, it records the
    slot's first non-finite step.  With ``ACTIONS``, ``allowed[r]`` is whether
    the slot has no record (this step or earlier).
    """

    r = tl.program_id(0)
    slot = tl.load(index_ptr + 1 * R + r)
    step = tl.load(index_ptr + 2 * R + r)
    delivered = tl.load(ok_ptr) != 0
    flags = tl.zeros([BLOCK_D], dtype=tl.int32)
    for start in range(0, D, BLOCK_D):
        offsets = start + tl.arange(0, BLOCK_D)
        # Masked lanes load 0.0, which is finite.
        hidden = tl.load(hidden_ptr + r * hidden_stride + offsets, mask=offsets < D, other=0.0)
        flags = flags | _not_finite(hidden.to(tl.float32)).to(tl.int32)
    bad = tl.max(flags, axis=0)
    for start in range(0, F, BLOCK_F):
        offsets = start + tl.arange(0, BLOCK_F)
        scores = tl.load(scores_ptr + r * scores_stride + offsets, mask=offsets < F, other=0.0)
        bad = bad | tl.max(_not_finite(scores).to(tl.int32), axis=0)
    bad = delivered & (bad > 0)
    if ACTIONS:
        # Every row here is a real row (slot >= 0): read its earlier record.
        recorded = _record_nonfinite(nonfinite_ptr, slot, step, bad, slot >= 0)
        tl.store(allowed_ptr + r, recorded == 0)
    else:
        _record_nonfinite(nonfinite_ptr, slot, step, bad, bad)
