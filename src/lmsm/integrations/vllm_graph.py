"""Graph-compatible observation for the vLLM integration.

The Python forward hook used by the eager integration does not run inside
torch.compile graphs or during CUDA-graph replay.  This module replaces it with
a custom operator, ``lmsm::capture_rows``, placed at the declared observation
point.  The operator copies the activation rows of the current forward into a
persistent buffer and copies a device-side step epoch into a device-side
``captured_epoch``.  Both are GPU kernels on tensors with fixed addresses.
CUDA-graph capture therefore records them, and every replay repeats them, in
full decode graphs, in piecewise graphs, and in compiled forwards that run
without graphs.

The host writes the expected epoch before each forward with an asynchronous
``fill_``.  After the forward the runtime compares ``captured_epoch`` with the
expected value.  A mismatch means the capture did not run, and the execution
contract fails closed the requests that needed this forward's observation.

Two observation points are supported.

* Module input (``pre:<module>`` keys).  ``lmsm::capture_rows`` reads the
  module's first argument.
* Residual stream after a decoder layer (module-output keys on a decoder
  layer, for example Gemma Scope's ``model.language_model.layers.22``).  vLLM's
  decoder layers return ``(hidden_states, residual)`` and defer the residual
  add to the next layer's fused add-norm.  The tensor that a Hugging Face
  forward hook sees as the layer's output, ``residual + hidden_states``, is
  therefore never materialized by the layer itself.
  ``lmsm::capture_residual_sum`` computes that sum into the persistent buffer
  and stamps the epoch.  The sum is one elementwise add in the model dtype,
  with the same rounding as the next layer's add.
"""

from __future__ import annotations

from typing import Any

import torch


CAPTURE_OP_NAMESPACE = "lmsm"
CAPTURE_OP_NAME = "capture_rows"
CAPTURE_SUM_OP_NAME = "capture_residual_sum"
EPOCH_BUFFER_NAME = "lmsm_captured_epoch"

POINT_MODULE_INPUT = "module_input"
POINT_RESIDUAL_STREAM = "residual_stream_after_layer"

# vLLM decoder-layer classes whose forward returns ``(hidden_states, residual)``
# with the residual add deferred to the next layer, verified against the
# Hugging Face module output (residual + hidden_states).  Other classes are
# refused for module-output keys rather than observed with unverified semantics.
RESIDUAL_DEFERRED_DECODER_LAYERS = frozenset({"Gemma3DecoderLayer"})


class UnsupportedObservationPoint(ValueError):
    """The declared activation key has no verified observation point in vLLM."""

    requirement = "verified_observation_point"


class GraphCaptureState:
    """Persistent device buffers shared by the capture op and the runtime."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.buffer: torch.Tensor | None = None
        self.step_epoch: torch.Tensor | None = None
        self.captured_epoch: torch.Tensor | None = None
        self.module_name: str | None = None
        self.point: str = POINT_MODULE_INPUT
        self.max_tokens = 0
        self.width: int | None = None
        # Python-side calls of the op implementation.  The implementation runs
        # in Python only outside graph replay (profiling, capture, compiled
        # forwards without a graph).  A nonzero count proves that the op is in
        # the compiled model.  It does not count replays.
        self.impl_calls = 0
        self.observed_width: int | None = None
        self.max_rows_seen = 0
        self.disabled_for_test = False
        self.remote_installed = False

    @property
    def installed(self) -> bool:
        return self.buffer is not None and self.module_name is not None

    def allocate(
        self,
        max_tokens: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_tokens = int(max_tokens)
        self.width = int(width)
        self.buffer = torch.zeros((self.max_tokens, self.width), dtype=dtype, device=device)
        self.step_epoch = torch.zeros((1,), dtype=torch.int64, device=device)

    def set_step_epoch(self, epoch: int) -> None:
        """Enqueue the expected epoch for the next forward (no host sync)."""

        assert self.step_epoch is not None
        self.step_epoch.fill_(int(epoch))


CAPTURE = GraphCaptureState()
_LIBRARY: Any | None = None


def _capture_rows_impl(x: torch.Tensor, captured_epoch: torch.Tensor) -> None:
    state = CAPTURE
    state.impl_calls += 1
    rows = int(x.shape[0])
    state.observed_width = int(x.shape[-1])
    state.max_rows_seen = max(state.max_rows_seen, rows)
    if state.buffer is None or state.step_epoch is None:
        raise RuntimeError("LMSM graph capture buffers are not allocated")
    if rows > state.max_tokens:
        raise RuntimeError(
            f"LMSM graph capture received {rows} rows; the persistent buffer "
            f"holds {state.max_tokens}"
        )
    state.buffer[:rows].copy_(x)
    captured_epoch.copy_(state.step_epoch)


def _capture_rows_fake(x: torch.Tensor, captured_epoch: torch.Tensor) -> None:
    del x, captured_epoch
    return None


def _capture_residual_sum_impl(
    hidden: torch.Tensor, residual: torch.Tensor, captured_epoch: torch.Tensor
) -> None:
    state = CAPTURE
    state.impl_calls += 1
    rows = int(hidden.shape[0])
    state.observed_width = int(hidden.shape[-1])
    state.max_rows_seen = max(state.max_rows_seen, rows)
    if state.buffer is None or state.step_epoch is None:
        raise RuntimeError("LMSM graph capture buffers are not allocated")
    if rows > state.max_tokens:
        raise RuntimeError(
            f"LMSM graph capture received {rows} rows; the persistent buffer "
            f"holds {state.max_tokens}"
        )
    # The residual stream after the layer, as the next layer's add-norm (and a
    # Hugging Face forward hook on the layer) sees it.
    torch.add(hidden, residual, out=state.buffer[:rows])
    captured_epoch.copy_(state.step_epoch)


def _capture_residual_sum_fake(
    hidden: torch.Tensor, residual: torch.Tensor, captured_epoch: torch.Tensor
) -> None:
    del hidden, residual, captured_epoch
    return None


def register_capture_op() -> None:
    """Register ``lmsm::capture_rows`` once per process."""

    global _LIBRARY
    if _LIBRARY is not None:
        return
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op

    library = Library(CAPTURE_OP_NAMESPACE, "FRAGMENT")
    direct_register_custom_op(
        op_name=CAPTURE_OP_NAME,
        op_func=_capture_rows_impl,
        mutates_args=["captured_epoch"],
        fake_impl=_capture_rows_fake,
        target_lib=library,
    )
    direct_register_custom_op(
        op_name=CAPTURE_SUM_OP_NAME,
        op_func=_capture_residual_sum_impl,
        mutates_args=["captured_epoch"],
        fake_impl=_capture_residual_sum_fake,
        target_lib=library,
    )
    from vllm.platforms import current_platform

    if current_platform.dispatch_key != "CPU":
        # CPU kernels for tests and CPU-only tools.
        library.impl(CAPTURE_OP_NAME, _capture_rows_impl, "CPU")
        library.impl(CAPTURE_SUM_OP_NAME, _capture_residual_sum_impl, "CPU")
    _LIBRARY = library


def resolve_observation_point(model: torch.nn.Module, activation_key: str) -> tuple[str, str]:
    """Map a profile's activation key to ``(point, vLLM module name)``.

    ``pre:<name>`` observes the module input.  Any other key observes the
    module output, which vLLM supports for the residual stream after a decoder
    layer (``RESIDUAL_DEFERRED_DECODER_LAYERS``).  Keys use Hugging Face module
    names.  A name that is not a vLLM module name is translated with the
    model's own checkpoint-name mapper (``hf_to_vllm_mapper``, for example
    Gemma 3's ``model.language_model.`` -> ``language_model.model.``).  The
    result must name exactly one module.
    """

    if activation_key.startswith("pre:"):
        point, declared = POINT_MODULE_INPUT, activation_key[len("pre:") :]
    else:
        point, declared = POINT_RESIDUAL_STREAM, activation_key
    modules = dict(model.named_modules())
    name = declared
    if name not in modules:
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        mapped = mapper.apply_list([declared]) if mapper is not None else []
        if len(mapped) != 1 or mapped[0] not in modules:
            raise UnsupportedObservationPoint(
                f"{UnsupportedObservationPoint.requirement}: no vLLM module for "
                f"{declared!r} (mapped: {mapped!r})"
            )
        name = mapped[0]
    if point == POINT_RESIDUAL_STREAM:
        layer_class = type(modules[name]).__name__
        if layer_class not in RESIDUAL_DEFERRED_DECODER_LAYERS:
            raise UnsupportedObservationPoint(
                f"{UnsupportedObservationPoint.requirement}: module-output key "
                f"{declared!r} resolves to {name!r} ({layer_class}); module "
                "outputs are supported for the residual stream after the decoder "
                f"layers {sorted(RESIDUAL_DEFERRED_DECODER_LAYERS)}"
            )
    return point, name


def install_graph_capture(
    model: torch.nn.Module,
    module_name: str,
    *,
    max_tokens: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    disabled_for_test: bool = False,
    point: str = POINT_MODULE_INPUT,
) -> torch.nn.Module:
    """Place the capture op at ``point`` of ``module_name``.

    The module's class is swapped for a subclass with a new ``forward``.  For
    the module input, it calls the op on its first argument and then runs the
    original forward.  For the residual stream after a decoder layer, it runs
    the original forward and calls the sum op on the returned
    ``(hidden_states, residual)``.  A class swap, rather than a hook or an
    instance attribute, keeps the call visible to Dynamo's full-graph trace.  ``disabled_for_test`` installs
    everything except the op call, which reproduces a deployment whose
    compiled graph lacks the capture (the execution check must then fail
    every request).
    """

    matches = [module for name, module in model.named_modules() if name == module_name]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {module_name!r}, found {len(matches)}")
    module = matches[0]
    register_capture_op()
    CAPTURE.allocate(max_tokens, width, dtype, device)
    if point not in (POINT_MODULE_INPUT, POINT_RESIDUAL_STREAM):
        raise ValueError(f"unknown observation point {point!r}")
    CAPTURE.module_name = module_name
    CAPTURE.point = point
    CAPTURE.disabled_for_test = bool(disabled_for_test)
    epoch = torch.full((1,), -1, dtype=torch.int64, device=device)
    module.register_buffer(EPOCH_BUFFER_NAME, epoch, persistent=False)
    CAPTURE.captured_epoch = getattr(module, EPOCH_BUFFER_NAME)

    original_class = type(module)
    namespace = getattr(torch.ops, CAPTURE_OP_NAMESPACE)
    capture_op = getattr(namespace, CAPTURE_OP_NAME)
    capture_sum_op = getattr(namespace, CAPTURE_SUM_OP_NAME)

    if disabled_for_test:

        class _ObservedModule(original_class):  # type: ignore[misc, valid-type]
            pass

    elif point == POINT_RESIDUAL_STREAM:

        class _ObservedModule(original_class):  # type: ignore[misc, valid-type]
            def forward(self, *args, **kwargs):  # noqa: D401 - module forward
                output = super().forward(*args, **kwargs)
                capture_sum_op(output[0], output[1], getattr(self, EPOCH_BUFFER_NAME))
                return output

    else:

        class _ObservedModule(original_class):  # type: ignore[misc, valid-type]
            def forward(self, x, *args, **kwargs):  # noqa: D401 - module forward
                capture_op(x, getattr(self, EPOCH_BUFFER_NAME))
                return super().forward(x, *args, **kwargs)

    _ObservedModule.__name__ = f"LMSMObserved{original_class.__name__}"
    _ObservedModule.__qualname__ = _ObservedModule.__name__
    module.__class__ = _ObservedModule
    return module


def capture_ok(expected_epoch: int) -> torch.Tensor:
    """Device bool: did the capture of the forward with ``expected_epoch`` run?"""

    assert CAPTURE.captured_epoch is not None
    return CAPTURE.captured_epoch == int(expected_epoch)


__all__ = [
    "CAPTURE",
    "CAPTURE_OP_NAME",
    "CAPTURE_OP_NAMESPACE",
    "CAPTURE_SUM_OP_NAME",
    "GraphCaptureState",
    "POINT_MODULE_INPUT",
    "POINT_RESIDUAL_STREAM",
    "RESIDUAL_DEFERRED_DECODER_LAYERS",
    "UnsupportedObservationPoint",
    "capture_ok",
    "install_graph_capture",
    "register_capture_op",
    "resolve_observation_point",
]
