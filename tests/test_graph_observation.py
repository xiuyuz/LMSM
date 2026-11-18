"""CPU tests for graph-captured observation and device-mode host bookkeeping."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmsm.backends.dense_probe import CheckpointRule, DenseProbe
from lmsm.execution_contract import (
    COMPLETED,
    EXECUTION_FAILED,
    assign_execution_state,
    check_engine_arguments,
    check_vllm_config,
)
from lmsm.integrations import vllm as runtime
from lmsm.loaders.profile_loader import load_profile
from lmsm.policy_evaluator import AnytimePolicyEvaluator, CheckpointPolicyEvaluator


ROOT = Path(__file__).resolve().parents[1]
GOOD_ENV = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
GOOD_ARGS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "enforce_eager": False,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
    "async_scheduling": False,
}


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    runtime.STATE.reset()
    from lmsm.integrations.vllm_graph import CAPTURE

    CAPTURE.reset()


def _compiled_config(**overrides):
    values = {"prefix": False, "chunked": False}
    values.update(overrides)
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="VLLM_COMPILE", value=3),
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE", value=(2, 1)),
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=values["prefix"]),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=values["chunked"], async_scheduling=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
    )


def test_graph_observation_allows_compiled_execution_and_nothing_else():
    hook = check_engine_arguments(GOOD_ARGS, GOOD_ENV, "0.19.0")
    assert [v.requirement for v in hook] == ["eager_execution"]
    assert check_engine_arguments(GOOD_ARGS, GOOD_ENV, "0.19.0", observation_mode="graph") == []
    defaults = check_engine_arguments(
        {"enforce_eager": False}, GOOD_ENV, "0.19.0", observation_mode="graph"
    )
    assert [v.requirement for v in defaults] == [
        "prefix_caching_off",
        "chunked_prefill_off",
        "async_scheduling_off",
    ]
    assert [v.requirement for v in check_vllm_config(_compiled_config(), GOOD_ENV, "0.19.0")] == [
        "eager_execution"
    ]
    assert check_vllm_config(_compiled_config(), GOOD_ENV, "0.19.0", observation_mode="graph") == []
    still_refused = check_vllm_config(
        _compiled_config(prefix=True), {}, "0.19.0", observation_mode="graph"
    )
    assert sorted(v.requirement for v in still_refused) == [
        "in_process_engine_core",
        "prefix_caching_off",
    ]
    with pytest.raises(ValueError):
        check_engine_arguments(GOOD_ARGS, GOOD_ENV, "0.19.0", observation_mode="trace")


def test_capture_op_records_rows_and_epoch_and_survives_dynamo():
    vllm_graph = pytest.importorskip("lmsm.integrations.vllm_graph")
    pytest.importorskip("vllm")

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.mlp(x) * 2

    # Seeded parameters and inputs from a local generator (independent of the
    # global RNG state), and an unwrapped Linear with the same parameters as
    # the reference.  The observed module runs the original Linear.forward, so
    # both sides take the same arithmetic path and agree exactly.  A
    # hand-written weight @ x.T + bias would round differently from Linear's
    # fused addmm near zero.
    generator = torch.Generator().manual_seed(20261114)
    model = Inner()
    with torch.no_grad():
        model.mlp.weight.copy_(torch.randn(4, 4, generator=generator))
        model.mlp.bias.copy_(torch.randn(4, generator=generator))
    reference = torch.nn.Linear(4, 4)
    reference.load_state_dict(model.mlp.state_dict())
    capture = vllm_graph.CAPTURE
    capture.reset()
    vllm_graph.install_graph_capture(
        model, "mlp", max_tokens=8, width=4, dtype=torch.float32, device=torch.device("cpu")
    )
    assert type(model.mlp).__name__ == "LMSMObservedLinear"
    x = torch.randn(3, 4, generator=generator)
    capture.set_step_epoch(5)
    out = model(x)
    assert torch.equal(capture.buffer[:3], x)
    assert int(capture.captured_epoch.item()) == 5
    assert torch.equal(out, reference(x) * 2)
    assert bool(vllm_graph.capture_ok(5).item()) and not bool(vllm_graph.capture_ok(6).item())

    for backend in ("eager", "aot_eager"):
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        y = x + 1
        capture.set_step_epoch(9)
        compiled(y)
        assert torch.equal(capture.buffer[:3], y), backend
        assert int(capture.captured_epoch.item()) == 9, backend
    capture.reset()


def _profile(name):
    profile = load_profile(ROOT / f"profiles/{name}.yaml")
    binding = replace(
        profile.backend_bindings[0],
        config={**profile.backend_bindings[0].config, "hidden_width": 2},
    )
    return replace(profile, backend_bindings=(binding,))


def _probe(target, width=2, threshold=0.5):
    return DenseProbe(
        target_id=target,
        mean=np.zeros(width),
        scale=np.ones(width),
        coef=np.ones(width),
        intercept=np.float64(0.0),
        threshold=np.float64(threshold),
    )


def _device_state(policy="checkpoint", rows=4):
    profile = _profile(f"lmsm_{policy}")
    if policy == "checkpoint":
        rule = CheckpointRule(
            target_id="chemical_biological",
            prompt=_probe("chemical_biological"),
            steps_1_32=_probe("chemical_biological"),
            steps_33_64=_probe("chemical_biological"),
            fusion=_probe("chemical_biological", width=3, threshold=2.0),
        )
        evaluator = CheckpointPolicyEvaluator([rule], profile)
    else:
        evaluator = AnytimePolicyEvaluator([_probe("chemical_biological")], profile)
    state = runtime.STATE
    state.configure(evaluator, [91, 92], observation_mode="graph", evaluation_mode="device")
    state.eos_token_id = 3
    state.init_device(max_rows=rows, max_steps=128, device=torch.device("cpu"))
    from lmsm.integrations.vllm_graph import CAPTURE

    CAPTURE.reset()
    CAPTURE.allocate(16, 2, torch.float32, torch.device("cpu"))
    state.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_tokens_no_spec=np.zeros(rows, dtype=np.int64),
            num_prompt_tokens=np.zeros(rows, dtype=np.int64),
            num_computed_tokens_cpu=np.zeros(rows, dtype=np.int64),
        )
    )
    return state


def _params(request_id):
    return SimpleNamespace(extra_args={runtime.REQUEST_ID_EXTRA_ARG: request_id})


def _update(*, batch_size, removed=(), added=(), moved=()):
    return SimpleNamespace(
        batch_size=batch_size, removed=list(removed), added=list(added), moved=list(moved)
    )


def _sync_batch(state, outputs_by_row, computed=None):
    batch = state.runner.input_batch
    for row, (prompt_len, outputs) in outputs_by_row.items():
        batch.num_prompt_tokens[row] = prompt_len
        batch.num_tokens_no_spec[row] = prompt_len + len(outputs)
        batch.num_computed_tokens_cpu[row] = (
            computed[row] if computed is not None else prompt_len + max(len(outputs) - 1, 0)
            if outputs else 0
        )


def _check_slot_map(state):
    for row, request in state.rows.items():
        slot = int(state.slot_of_row[row])
        assert slot >= 0
        assert state.slot_request[slot] == request.request_id
        assert state.slot_state[slot] is request
    live = {int(state.slot_of_row[row]) for row in state.rows}
    assert len(live) == len(state.rows)
    assert not live & set(state.free_slots)


def test_device_mode_slots_follow_requests_through_moves_swaps_and_reuse():
    state = _device_state("anytime")
    outs = {name: [] for name in "abcde"}
    state.update_batch(
        _update(
            batch_size=3,
            added=[(0, _params("a"), [1, 2], outs["a"]), (1, _params("b"), [1], outs["b"]),
                   (2, _params("c"), [1, 2, 3], outs["c"])],
        )
    )
    _check_slot_map(state)
    slot_a = state.rows[0].slot
    for name in "abc":
        outs[name].append(7)
    MOVE = SimpleNamespace(name="UNIDIRECTIONAL")
    SWAP = SimpleNamespace(name="SWAP")
    # b finishes, condense moves c from row 2 into row 1, and d is admitted at row 2.
    state.update_batch(
        _update(batch_size=3, removed=[1], added=[(2, _params("d"), [1, 2], outs["d"])],
                moved=[])
    )
    _check_slot_map(state)
    assert "b" in state.device_archive
    state.update_batch(_update(batch_size=3, moved=[(0, 2, SWAP)]))
    _check_slot_map(state)
    assert state.rows[2].request_id == "a" and state.rows[2].slot == slot_a
    # A departure hidden by condense: e moves onto the row of d.
    state.update_batch(_update(batch_size=3, added=[(1, _params("e"), [1], outs["e"])]))
    state.update_batch(_update(batch_size=2, moved=[(2, 1, MOVE)]))
    _check_slot_map(state)
    assert "e" in state.device_archive and state.rows[1].request_id == "a"
    assert len(state.free_slots) == state.max_rows - len(state.rows)


def test_device_planning_matches_scalar_eligibility_and_final_token_positions():
    state = _device_state("checkpoint")
    prompts = {"a": [1, 2, 3], "b": [4, 5], "c": [6, 7]}
    outs = {name: [] for name in prompts}
    state.update_batch(
        _update(
            batch_size=3,
            added=[(row, _params(name), prompts[name], outs[name]) for row, name in enumerate(prompts)],
        )
    )
    # Forward 1: a and b complete their prompts, and c's prompt is chunked (1 of 2).
    _sync_batch(state, {0: (3, outs["a"]), 1: (2, outs["b"]), 2: (2, outs["c"])}, computed=[0, 0, 0])
    state.begin_forward([3, 2, 1])
    pending = state._plan_device_step()
    assert list(pending.rows) == [0, 1]
    assert list(pending.tokens) == [2, 4]  # cumsum(counts) - 1 for the eligible rows
    assert list(pending.steps) == [0, 0]
    assert pending.flags.any_prompt and not pending.flags.any_evaluation
    assert not state.accounts["c"].failed  # a partial chunk processes no declared position
    # Forward 2: a and b decode step 1, and c completes its prompt (step 0).
    for name in "ab":
        outs[name].append(9)
    _sync_batch(state, {0: (3, outs["a"]), 1: (2, outs["b"]), 2: (2, outs["c"])}, computed=[3, 2, 1])
    state.begin_forward([1, 1, 1])
    pending = state._plan_device_step()
    assert list(pending.rows) == [0, 1, 2] and list(pending.steps) == [1, 1, 0]
    assert list(pending.tokens) == [0, 1, 2]
    # Decode forwards: the step index comes from the per-request forward counter.
    for step in range(2, 66):
        for name in "abc":
            outs[name].append(9)
        _sync_batch(state, {0: (3, outs["a"]), 1: (2, outs["b"]), 2: (2, outs["c"])})
        state.begin_forward([1, 1, 1])
        pending = state._plan_device_step()
        expected = [s for s in (step, step, step - 1) if s <= 64]
        assert list(pending.steps) == expected
        assert pending.flags.any_window_1_end == (32 in expected)
        assert pending.flags.any_evaluation == (64 in expected)

    required = state.requirements
    steps = np.arange(0, 80)
    for decision in (-1, 10, 64):
        vector = state._requires_observation_vector(steps, np.full(steps.shape, decision))
        scalar = [
            required.requires_observation(int(s), None if decision < 0 else decision) for s in steps
        ]
        assert vector.tolist() == scalar


def test_device_accounts_read_the_delivered_observation_map():
    state = _device_state("anytime")
    outs = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [1, 2], outs)]))
    outs.extend([5, 6, 7, 8])
    record = {
        "observed": np.array([0, 1, 1, 0, 1], dtype=np.uint8),  # step 3 never delivered
        "policy_scores": {}, "policy_score_steps": {}, "policy_actions": {},
        "evaluation_error": None,
    }
    state._store_archive("a", record)
    state._materialize_device_account("a")
    outcome = assign_execution_state(state.requirements, state.accounts["a"], len(outs))
    assert outcome.state == EXECUTION_FAILED and outcome.missing_observation_steps == [3]

    record = dict(record, observed=np.array([0, 1, 1, 1, 1], dtype=np.uint8), materialized=False)
    state.accounts["a"].observed = None
    state._store_archive("a", record)
    state._materialize_device_account("a")
    outcome = assign_execution_state(state.requirements, state.accounts["a"], len(outs))
    assert outcome.state == COMPLETED


def test_device_checkpoint_account_requires_the_fusion_evaluation_itself():
    """A delivered observation at the checkpoint does not count as its evaluation."""

    state = _device_state("checkpoint")
    outs = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [1, 2], outs)]))
    outs.extend(range(5, 5 + 65))  # 65 generated tokens, so step 64 was processed
    base = {
        "observed": np.ones(66, dtype=np.uint8),
        "policy_scores": {"prompt": {}, "steps_1_32": {}, "steps_33_64": {}, "fusion": {}},
        "policy_score_steps": {}, "policy_actions": {},
        "evaluation_error": None,
    }
    state._store_archive("a", dict(base, evaluated_points=[]))
    state._materialize_device_account("a")
    outcome = assign_execution_state(state.requirements, state.accounts["a"], len(outs))
    assert outcome.state == EXECUTION_FAILED and outcome.missing_evaluation_steps == [64]

    state.accounts["a"].observed = None
    state._store_archive("a", dict(base, evaluated_points=[64], materialized=False))
    state._materialize_device_account("a")
    outcome = assign_execution_state(state.requirements, state.accounts["a"], len(outs))
    assert outcome.state == COMPLETED


def test_device_selected_feature_engine_matches_the_host_transcoder_path():
    from lmsm.backends.shared import SharedFeatureBackend  # noqa: F401
    from lmsm.build import build_shared_feature_backends
    from lmsm.aggregators.max_score import MaxScoreAggregator
    from lmsm.policies.threshold import ThresholdPolicy
    from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator
    from lmsm.loaders.sae_loader import LinearTranscoderAdapter
    from lmsm.integrations.vllm_device import DeviceSharedFeatureEngine, StepFlags

    torch.manual_seed(0)
    profile = load_profile(ROOT / "profiles/lmsm_transcoder.yaml")
    binding = profile.backend_bindings[0]
    width = 16
    features = max(
        int(i) for rule in profile.active_rules for i in rule.condition.config["feature_indices"]
    ) + 1
    artifact = LinearTranscoderAdapter(
        W_enc=(torch.randn(features, width) * 0.5).to(torch.bfloat16),
        b_enc=(torch.randn(features) * 0.1).to(torch.bfloat16),
        W_dec=torch.zeros(1, width, dtype=torch.bfloat16),
        b_dec=torch.zeros(width, dtype=torch.bfloat16),
    )
    shared, bound = build_shared_feature_backends(profile, binding, artifact, selected=True)
    host = SharedFeaturePolicyEvaluator(
        shared, bound, MaxScoreAggregator(), ThresholdPolicy(actions_enabled=True), profile
    )
    engine = DeviceSharedFeatureEngine(host, 4, 4, 64, torch.device("cpu"))
    buffer = torch.zeros((64, width), dtype=torch.bfloat16)
    epoch = torch.zeros((1,), dtype=torch.int64)
    engine.bind_capture(buffer, epoch)
    ids = ["a", "b", "c", "d"]
    for request_id in ids:
        host.add(request_id)
    engine.reset_slots(torch.arange(4))
    host_actions, device_actions = {}, {}
    actioned = set()
    for step in range(12):
        live = [i for i, r in enumerate(ids) if r not in actioned]
        if not live:
            break
        rows = (torch.randn(len(live), width) * 3).to(torch.bfloat16)
        buffer[: len(live)] = rows
        epoch.fill_(step + 1)
        decisions = host.evaluate([ids[i] for i in live], buffer[: len(live)], [step] * len(live))
        engine.load(
            np.asarray(live), np.asarray(live), np.full(len(live), step), np.arange(len(live)), step + 1
        )
        produced = engine.run(StepFlags())
        assert produced
        result = engine.result[: len(live)]
        for position, i in enumerate(live):
            host_hit = decisions[position].action != "allow"
            device_hit = bool(result[position, 0] > 0.5)
            assert host_hit == device_hit, (step, ids[i])
            if host_hit:
                rule_index = int(result[position, 1])
                assert engine.target_ids[rule_index] == decisions[position].params["target_id"]
                assert float(result[position, 2]) == decisions[position].params["score"]
                assert engine.threshold_for(rule_index) == decisions[position].params["threshold"]
                host_actions[ids[i]] = (step, decisions[position].params["rule_id"])
                device_actions[ids[i]] = (step, profile.rule_for_target(engine.target_ids[rule_index]).rule_id)
                actioned.add(ids[i])
    assert host_actions == device_actions and host_actions
    records = engine.archive([0, 1, 2, 3], [12] * 4)
    for slot, request_id in enumerate(ids):
        assert records[slot]["feature_maxima"] == host.feature_maxima_for(request_id)


def test_device_evaluation_refuses_non_elementwise_or_unknown_feature_encoders():
    from lmsm.aggregators.max_score import MaxScoreAggregator
    from lmsm.build import build_shared_feature_backends
    from lmsm.integrations.vllm_device import UnsupportedDeviceBackend, check_device_evaluator
    from lmsm.policies.threshold import ThresholdPolicy
    from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator

    class TopKSAE:  # a full encoder whose activation is not elementwise
        def encode(self, x):
            return x

    profile = load_profile(ROOT / "profiles/lmsm_transcoder.yaml")
    shared, bound = build_shared_feature_backends(
        profile, profile.backend_bindings[0], TopKSAE(), selected=True
    )
    evaluator = SharedFeaturePolicyEvaluator(
        shared, bound, MaxScoreAggregator(), ThresholdPolicy(), profile
    )
    with pytest.raises(UnsupportedDeviceBackend, match="elementwise_selected_feature_encoder"):
        check_device_evaluator(evaluator)
    with pytest.raises(UnsupportedDeviceBackend):
        runtime.STATE.configure(
            evaluator, [1], observation_mode="graph", evaluation_mode="device"
        )


def _fake_decoder_layer_class(name):
    """A vLLM-style decoder layer: returns (hidden_states, residual), add deferred."""

    def forward(self, positions, hidden_states, residual):
        del positions
        if residual is None:
            residual = hidden_states
        else:
            residual = hidden_states + residual
        return self.proj(hidden_states) * 0.5, residual

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.proj = torch.nn.Linear(4, 4)

    return type(name, (torch.nn.Module,), {"__init__": __init__, "forward": forward})


class _FakeMapper:
    def __init__(self, prefixes):
        self.prefixes = prefixes

    def apply_list(self, values):
        out = []
        for value in values:
            for old, new in self.prefixes.items():
                if value.startswith(old):
                    out.append(new + value[len(old):])
                    break
        return out


def _fake_gemma_model(layer_class_name="Gemma3DecoderLayer"):
    layer_class = _fake_decoder_layer_class(layer_class_name)

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([layer_class() for _ in range(3)])

        def forward(self, x):
            residual = None
            for layer in self.layers:
                x, residual = layer(None, x, residual)
            return x + residual

    class Causal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Stack()

        def forward(self, x):
            return self.model(x)

    class Conditional(torch.nn.Module):
        hf_to_vllm_mapper = _FakeMapper({"model.language_model.": "language_model.model."})

        def __init__(self):
            super().__init__()
            self.language_model = Causal()

        def forward(self, x):
            return self.language_model(x)

    return Conditional()


def test_module_output_keys_resolve_through_the_checkpoint_mapper_to_verified_layers():
    vllm_graph = pytest.importorskip("lmsm.integrations.vllm_graph")
    model = _fake_gemma_model()
    assert vllm_graph.resolve_observation_point(model, "model.language_model.layers.1") == (
        vllm_graph.POINT_RESIDUAL_STREAM,
        "language_model.model.layers.1",
    )
    # vLLM names resolve to themselves, and module-input keys keep their module.
    assert vllm_graph.resolve_observation_point(model, "language_model.model.layers.2") == (
        vllm_graph.POINT_RESIDUAL_STREAM,
        "language_model.model.layers.2",
    )
    assert vllm_graph.resolve_observation_point(model, "pre:model.language_model.layers.0.proj") == (
        vllm_graph.POINT_MODULE_INPUT,
        "language_model.model.layers.0.proj",
    )
    with pytest.raises(vllm_graph.UnsupportedObservationPoint, match="verified_observation_point"):
        vllm_graph.resolve_observation_point(model, "model.language_model.layers.9")
    # A module output that is not a verified decoder layer's residual stream is refused.
    with pytest.raises(vllm_graph.UnsupportedObservationPoint, match="verified_observation_point"):
        vllm_graph.resolve_observation_point(model, "model.language_model.layers.0.proj")
    with pytest.raises(vllm_graph.UnsupportedObservationPoint, match="Qwen3DecoderLayer"):
        vllm_graph.resolve_observation_point(
            _fake_gemma_model("Qwen3DecoderLayer"), "model.language_model.layers.1"
        )


def test_residual_stream_capture_matches_the_hf_layer_output_and_survives_dynamo():
    vllm_graph = pytest.importorskip("lmsm.integrations.vllm_graph")
    pytest.importorskip("vllm")
    torch.manual_seed(0)
    model = _fake_gemma_model()
    capture = vllm_graph.CAPTURE
    capture.reset()
    name = "language_model.model.layers.1"
    layer = dict(model.named_modules())[name]
    x = torch.randn(3, 4)

    # Reference: the HF-style layer output (residual + hidden) of layer 1.
    with torch.no_grad():
        h0, r0 = model.language_model.model.layers[0](None, x, None)
        h1, r1 = layer(None, h0, r0)
        expected = r1 + h1
        reference_out = model(x)

    vllm_graph.install_graph_capture(
        model,
        name,
        max_tokens=8,
        width=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        point=vllm_graph.POINT_RESIDUAL_STREAM,
    )
    assert type(layer).__name__ == "LMSMObservedGemma3DecoderLayer"
    assert capture.point == vllm_graph.POINT_RESIDUAL_STREAM
    capture.set_step_epoch(4)
    with torch.no_grad():
        out = model(x)
    assert torch.equal(out, reference_out)  # the model's output is unchanged
    assert torch.equal(capture.buffer[:3], expected)
    assert int(capture.captured_epoch.item()) == 4

    # The eager hook path computes the same sum.
    seen = []
    original = runtime.STATE.capture
    runtime.STATE.capture = seen.append
    try:
        runtime._capture_residual_stream_output(layer, (), (h1, r1))
    finally:
        runtime.STATE.capture = original
    assert torch.equal(seen[0], expected)

    for backend in ("eager", "aot_eager"):
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        y = x * 2
        with torch.no_grad():
            g0, s0 = model.language_model.model.layers[0](None, y, None)
            g1, s1 = _fake_decoder_layer_class("Gemma3DecoderLayer").forward(layer, None, g0, s0)
        capture.set_step_epoch(7)
        with torch.no_grad():
            compiled(y)
        assert torch.equal(capture.buffer[:3], s1 + g1), backend
        assert int(capture.captured_epoch.item()) == 7, backend
    capture.reset()


def _jumprelu_sae(d_in, d_sae, *, apply_b_dec_to_input=False, normalize="none", seed=0):
    sae_lens = pytest.importorskip("sae_lens")
    from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig

    del sae_lens
    cfg = JumpReLUSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        apply_b_dec_to_input=apply_b_dec_to_input,
        normalize_activations=normalize,
    )
    sae = JumpReLUSAE(cfg)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        sae.W_enc.copy_(torch.randn(d_in, d_sae, generator=generator) * 40.0)
        sae.b_enc.copy_(torch.randn(d_sae, generator=generator) * 20.0)
        sae.threshold.copy_(torch.rand(d_sae, generator=generator) * 200.0)
        sae.b_dec.copy_(torch.randn(d_in, generator=generator))
    return sae


@pytest.mark.parametrize("apply_b_dec_to_input", [False, True])
def test_device_selected_feature_engine_matches_the_host_jumprelu_sae_path(apply_b_dec_to_input):
    from lmsm.aggregators.max_score import MaxScoreAggregator
    from lmsm.build import build_shared_feature_backends
    from lmsm.integrations.vllm_device import DeviceSharedFeatureEngine, StepFlags, _selected_encoder
    from lmsm.policies.threshold import ThresholdPolicy
    from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator

    torch.manual_seed(1)
    profile = load_profile(ROOT / "profiles/lmsm_sae.yaml")
    binding = profile.backend_bindings[0]
    width = 16
    features = max(
        int(i) for rule in profile.active_rules for i in rule.condition.config["feature_indices"]
    ) + 1
    sae = _jumprelu_sae(width, features, apply_b_dec_to_input=apply_b_dec_to_input)
    shared, bound = build_shared_feature_backends(profile, binding, sae, selected=True)
    assert not hasattr(sae, "encode_selected")  # the host path runs sae_lens's full encode
    host = SharedFeaturePolicyEvaluator(
        shared, bound, MaxScoreAggregator(), ThresholdPolicy(actions_enabled=True), profile
    )
    # The selected encoder is sae_lens's encode restricted to the union of features.
    encoder = _selected_encoder(sae, list(shared.feature_indices))
    probe = (torch.randn(32, width) * 4).to(torch.bfloat16)
    with torch.no_grad():
        full = sae.encode(probe)[:, list(shared.feature_indices)]
    selected = encoder.encode(probe)
    assert torch.allclose(selected, full, rtol=1e-6, atol=1e-4)
    assert torch.equal(selected > 0, full > 0)

    engine = DeviceSharedFeatureEngine(host, 4, 4, 64, torch.device("cpu"))
    buffer = torch.zeros((64, width), dtype=torch.bfloat16)
    epoch = torch.zeros((1,), dtype=torch.int64)
    engine.bind_capture(buffer, epoch)
    ids = ["a", "b", "c", "d"]
    for request_id in ids:
        host.add(request_id)
    engine.reset_slots(torch.arange(4))
    host_actions, device_actions = {}, {}
    actioned = set()
    for step in range(16):
        live = [i for i, r in enumerate(ids) if r not in actioned]
        if not live:
            break
        rows = (torch.randn(len(live), width) * 4).to(torch.bfloat16)
        buffer[: len(live)] = rows
        epoch.fill_(step + 1)
        with torch.no_grad():
            decisions = host.evaluate([ids[i] for i in live], buffer[: len(live)], [step] * len(live))
        engine.load(
            np.asarray(live), np.asarray(live), np.full(len(live), step), np.arange(len(live)), step + 1
        )
        assert engine.run(StepFlags())
        result = engine.result[: len(live)]
        for position, i in enumerate(live):
            host_hit = decisions[position].action != "allow"
            assert host_hit == bool(result[position, 0] > 0.5), (step, ids[i])
            if host_hit:
                rule_index = int(result[position, 1])
                assert engine.target_ids[rule_index] == decisions[position].params["target_id"]
                assert float(result[position, 2]) == decisions[position].params["score"]
                host_actions[ids[i]] = (step, decisions[position].params["rule_id"])
                device_actions[ids[i]] = (
                    step, profile.rule_for_target(engine.target_ids[rule_index]).rule_id
                )
                actioned.add(ids[i])
    assert host_actions == device_actions and host_actions
    records = engine.archive([0, 1, 2, 3], [16] * 4)
    for slot, request_id in enumerate(ids):
        device_maxima = records[slot]["feature_maxima"]
        host_maxima = host.feature_maxima_for(request_id)
        assert device_maxima.keys() == host_maxima.keys()
        for category, values in host_maxima.items():
            assert device_maxima[category].keys() == values.keys()
            for feature, value in values.items():
                assert device_maxima[category][feature] == pytest.approx(value, rel=1e-6, abs=1e-4)


def test_device_evaluation_refuses_sae_lens_encoders_it_cannot_reproduce():
    from lmsm.aggregators.max_score import MaxScoreAggregator
    from lmsm.build import build_shared_feature_backends
    from lmsm.integrations.vllm_device import UnsupportedDeviceBackend, check_device_evaluator
    from lmsm.policies.threshold import ThresholdPolicy
    from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator

    profile = load_profile(ROOT / "profiles/lmsm_sae.yaml")
    features = 222800

    def evaluator_for(sae):
        shared, bound = build_shared_feature_backends(
            profile, profile.backend_bindings[0], sae, selected=True
        )
        return SharedFeaturePolicyEvaluator(
            shared, bound, MaxScoreAggregator(), ThresholdPolicy(), profile
        )

    check_device_evaluator(evaluator_for(_jumprelu_sae(4, features)))
    with pytest.raises(UnsupportedDeviceBackend, match="normalization"):
        check_device_evaluator(evaluator_for(_jumprelu_sae(4, features, normalize="layer_norm")))
    hooked = _jumprelu_sae(4, features)
    hooked.hook_sae_acts_pre.register_forward_hook(lambda module, args, out: out * 2)
    with pytest.raises(UnsupportedDeviceBackend, match="hook_sae_acts_pre"):
        check_device_evaluator(evaluator_for(hooked))


# ---------------------------------------------------------------------------
# Fused Triton step kernels, run on the CPU by Triton's interpreter


def _in_triton_interpreter(check, *args):
    """Run ``check(*args)``, a function of this module, under Triton's CPU interpreter.

    The interpreter must be enabled before ``triton.language`` is first
    imported (its standard-library helpers are themselves ``@triton.jit``
    functions), and this test process has imported Triton already.  The check
    therefore runs in a child process started with ``TRITON_INTERPRET=1`` and
    no visible GPU.  The child hides vLLM, which the checks do not use, so
    ``lmsm.integrations.vllm`` imports quickly.
    """

    pytest.importorskip("triton")
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json, runpy, sys\n"
        "sys.modules['vllm'] = None\n"
        "module = runpy.run_path(sys.argv[1])\n"
        "module[sys.argv[2]](*json.loads(sys.argv[3]))\n"
    )
    path = [str(ROOT / "src")] + [p for p in [os.environ.get("PYTHONPATH")] if p]
    env = dict(
        os.environ,
        TRITON_INTERPRET="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(path),
        OMP_NUM_THREADS="1",  # the interpreter runs tiny numpy/torch ops
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, check, json.dumps(args)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _random_probe(rng, target, width, threshold):
    return DenseProbe(
        target_id=target,
        mean=rng.normal(size=width),
        scale=rng.uniform(0.5, 2.0, size=width),
        coef=rng.normal(size=width) / np.sqrt(width),
        intercept=np.float64(0.1),
        threshold=np.float64(threshold),
    )


def _dense_evaluator(policy, width, seed=0):
    """Three random rules of the profile's policy that read `width` values."""

    rng = np.random.default_rng(seed)
    profile = _profile(f"lmsm_{policy}")
    targets = [rule.target_id for rule in profile.active_rules][:3]
    if policy == "anytime":
        return AnytimePolicyEvaluator([_random_probe(rng, t, width, 0.6) for t in targets], profile)
    rules = [
        CheckpointRule(
            target_id=target,
            prompt=_random_probe(rng, target, width, 0.0),
            steps_1_32=_random_probe(rng, target, width, 0.0),
            steps_33_64=_random_probe(rng, target, width, 0.0),
            fusion=_random_probe(rng, target, 3, fusion_threshold),
        )
        # Fusion thresholds: always crossed, data-dependent, never crossed.
        for target, fusion_threshold in zip(targets, (-50.0, 0.0, 50.0))
    ]
    return CheckpointPolicyEvaluator(rules, profile)


def _dense_engine(evaluator, *, triton, rows=4, device="cpu", graphs=False):
    from lmsm.integrations.vllm_device import DeviceAnytimeEngine, DeviceCheckpointEngine

    cls = DeviceAnytimeEngine if isinstance(evaluator, AnytimePolicyEvaluator) else DeviceCheckpointEngine
    engine = cls(evaluator, rows, rows, 80, torch.device(device))
    engine.use_graphs, engine.use_triton = graphs, triton
    return engine


def _policy_steps(engine):
    if getattr(engine, "evaluation_step", None) is None:
        return list(range(1, 11))
    return list(range(0, engine.evaluation_step + 1))


def _run_dense_steps(engines, buffers, width, steps, *, live=3, seed=1, poison=None):
    """Give every engine the same activation rows and return per-step results.

    ``poison`` maps (row, step) to a value written into that row's activation.
    Rows are slots 0..live-1, and each engine's remaining row is padding.  An
    engine with ``use_graphs`` (CUDA) runs its captured step graphs.
    """

    from lmsm.integrations.vllm_device import StepFlags

    generator = torch.Generator().manual_seed(seed)
    epoch = torch.zeros((1,), dtype=torch.int64, device=buffers[0].device)
    for engine, buffer in zip(engines, buffers, strict=True):
        engine.bind_capture(buffer, epoch)
        engine.reset_slots(torch.arange(engine.num_slots, device=engine.device))
        engine.capture_graphs()  # no-op without use_graphs
    trace = [[] for _ in engines]
    for step in steps:
        values = (torch.randn(live, width, generator=generator) * 2).to(torch.bfloat16)
        for (row, at), value in (poison or {}).items():
            if at == step:
                values[row, width // 2] = value
        epoch.fill_(step + 1)
        flags = StepFlags(
            any_prompt=step == 0,
            any_window_1_end=step == getattr(engines[0], "split", None),
            any_evaluation=step == getattr(engines[0], "evaluation_step", step),
        )
        for engine, buffer, out in zip(engines, buffers, trace, strict=True):
            buffer[:live] = values
            engine.load(np.arange(live), np.arange(live), np.full(live, step), np.arange(live), step + 1)
            produced = engine.run(flags)
            out.append((produced, engine.result.clone(), engine.act_rows.clone(), bool(engine.ok.item())))
    return trace


def _assert_engines_agree(kernel, reference, kernel_trace, reference_trace):
    """Bitwise for sums, counts, maps and decisions, and probe scores to 1e-12."""

    from lmsm.integrations.vllm_device import DeviceAnytimeEngine

    if isinstance(kernel, DeviceAnytimeEngine):
        exact = ("observed", "nonfinite", "sums", "counts", "max_steps_tensor", "crossed")
        close = ("maxima",)
    else:
        exact = ("observed", "nonfinite", "window_sums", "window_counts", "done", "crossed", "errors")
        close = ("scores",)
    for name in exact:
        assert torch.equal(getattr(kernel, name), getattr(reference, name)), name
    for name in close:
        torch.testing.assert_close(getattr(kernel, name), getattr(reference, name), rtol=1e-12, atol=1e-12)
    for (k_produced, k_result, k_act, k_ok), (r_produced, r_result, r_act, r_ok) in zip(
        kernel_trace, reference_trace, strict=True
    ):
        assert (k_produced, k_ok) == (r_produced, r_ok)
        assert torch.equal(k_act, r_act)
        assert torch.equal(k_result[:, :2], r_result[:, :2])  # triggered, rule
        torch.testing.assert_close(k_result, r_result, rtol=1e-12, atol=1e-12)


KERNEL_BOUNDS_CASES = [
    ["anytime", 40, 16],  # 3 blocks, the last one partial
    ["anytime", 40, None],  # the default block (512) is wider than the row
    ["anytime", 64, 16],  # divides the width: every mask is all-true
    ["checkpoint", 300, None],  # 256 + 44: the second block is partial
    ["checkpoint", 512, None],  # divides the width
]


def _check_kernel_bounds(cases, device="cpu"):
    """Kernel engine against the PyTorch reference engine, case by case.

    The kernel engine's capture buffer and probe weights are views whose
    out-of-row memory holds NaN, so a read past a row would surface as NaN
    sums or scores.  Its per-slot sums sit before a sentinel row, so a write
    past the last slot would change the sentinel.  On CUDA both engines run
    their captured step graphs.
    """

    import os

    graphs = torch.device(device).type == "cuda"
    for policy, width, block in cases:
        if block is None:
            os.environ.pop("LMSM_ANYTIME_BLOCK_D", None)
        else:
            os.environ["LMSM_ANYTIME_BLOCK_D"] = str(block)
        evaluator = _dense_evaluator(policy, width)
        kernel = _dense_engine(evaluator, triton=True, device=device, graphs=graphs)
        reference = _dense_engine(evaluator, triton=False, device=device, graphs=graphs)
        pad = 32
        kernel_buffer = torch.full(
            (16, width + pad), float("nan"), dtype=torch.bfloat16, device=device
        )[:, :width]
        # Rows are finite, because CUDA graph warm-up runs padding rows, which
        # read row 0.  Only the memory past each row holds NaN.
        kernel_buffer.zero_()
        reference_buffer = torch.zeros((16, width), dtype=torch.bfloat16, device=device)
        sums_name = "sums" if policy == "anytime" else "window_sums"
        sums = getattr(kernel, sums_name)
        padded_sums = torch.full((sums.shape[0] + 1, width), 7.0, dtype=sums.dtype, device=device)
        padded_sums[:-1] = sums
        setattr(kernel, sums_name, padded_sums[:-1])
        if policy == "anytime":
            padded_weight = torch.full(
                (width + pad, kernel.weight.shape[1]),
                float("nan"),
                dtype=kernel.weight.dtype,
                device=device,
            )
            padded_weight[:width] = kernel.weight
            kernel.weight = padded_weight[:width]
        kernel_trace, reference_trace = _run_dense_steps(
            [kernel, reference], [kernel_buffer, reference_buffer], width, _policy_steps(kernel)
        )
        case = (policy, width, block)
        assert torch.equal(padded_sums[-1].cpu(), torch.full((width,), 7.0)), ("write past a row", case)
        assert not torch.isnan(getattr(kernel, sums_name)).any(), ("read past a row", case)
        _assert_engines_agree(kernel, reference, kernel_trace, reference_trace)
        assert any(bool(out[2].any()) for out in reference_trace), ("no step acted", case)


def test_triton_kernels_stay_inside_each_row_for_any_block_size():
    """The fused kernels mask every hidden-dimension access (Triton CPU interpreter).

    Block sizes that do not divide the hidden width read and write only their
    own row and agree with the PyTorch reference engine, as do block sizes
    that divide it (every mask all-true, the default configuration).
    """

    _in_triton_interpreter("_check_kernel_bounds", KERNEL_BOUNDS_CASES)


@pytest.mark.parametrize("policy", ["anytime", "checkpoint"])
def test_device_engine_refuses_a_capture_buffer_of_another_width(policy):
    """The kernels read the policy's width from each buffer row: it must be the row width."""

    engine = _dense_engine(_dense_evaluator(policy, 8), triton=False)
    epoch = torch.zeros((1,), dtype=torch.int64)
    with pytest.raises(ValueError, match="width"):
        engine.bind_capture(torch.zeros((16, 9), dtype=torch.bfloat16), epoch)
    engine.bind_capture(torch.zeros((16, 8), dtype=torch.bfloat16), epoch)


# ---------------------------------------------------------------------------
# Selected-feature policies: duplicate feature IDs


def _feature_profile(rule_configs, hidden_width=None):
    """The transcoder profile reduced to its first rules, reconfigured.

    ``rule_configs`` gives, per rule, the condition-config entries to replace.
    """

    profile = load_profile(ROOT / "profiles/lmsm_transcoder.yaml")
    if hidden_width is not None:
        binding = profile.backend_bindings[0]
        binding = replace(binding, config={**binding.config, "hidden_width": hidden_width})
        profile = replace(profile, backend_bindings=(binding,))
    rules = list(profile.rule_library.rules)
    for index, overrides in enumerate(rule_configs):
        rule = rules[index]
        config = {**rule.condition.config, **overrides}
        rules[index] = replace(rule, condition=replace(rule.condition, config=config))
    profile = replace(profile, rule_library=replace(profile.rule_library, rules=tuple(rules)))
    return profile.with_active_rules([rule.rule_id for rule in rules[: len(rule_configs)]])


def _feature_evaluator(profile, artifact):
    from lmsm.aggregators.max_score import MaxScoreAggregator
    from lmsm.build import build_shared_feature_backends
    from lmsm.policies.threshold import ThresholdPolicy
    from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator

    shared, bound = build_shared_feature_backends(
        profile, profile.backend_bindings[0], artifact, selected=True
    )
    return SharedFeaturePolicyEvaluator(
        shared, bound, MaxScoreAggregator(), ThresholdPolicy(actions_enabled=True), profile
    )


def _transcoder(features, width, seed=0, scale=0.5):
    from lmsm.loaders.sae_loader import LinearTranscoderAdapter

    generator = torch.Generator().manual_seed(seed)
    return LinearTranscoderAdapter(
        W_enc=(torch.randn(features, width, generator=generator) * scale).to(torch.bfloat16),
        b_enc=(torch.randn(features, generator=generator) * 0.1).to(torch.bfloat16),
        W_dec=torch.zeros(1, width, dtype=torch.bfloat16),
        b_dec=torch.zeros(width, dtype=torch.bfloat16),
    )


def test_device_evaluation_refuses_duplicate_feature_ids_and_loads_the_released_profiles():
    """A rule listing a feature twice is refused at startup with a named requirement.

    The host path counts both entries: here feature 0 has value 2, both
    entries have threshold 1, so the rule's fraction is 1.0 and it refuses.
    One device column per feature would have counted one of the two entries
    (0.5, allow).  The released transcoder and SAE profiles have no duplicate
    and are accepted.
    """

    from lmsm.integrations.vllm_device import (
        DeviceSharedFeatureEngine,
        DuplicateFeatureIds,
        UnsupportedDeviceBackend,
        check_device_evaluator,
    )
    from lmsm.loaders.sae_loader import LinearTranscoderAdapter

    profile = _feature_profile(
        [{"feature_indices": [0, 0], "feature_thresholds": [1.0, 1.0], "threshold": 1.0}]
    )
    artifact = LinearTranscoderAdapter(
        W_enc=torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        b_enc=torch.zeros(1, dtype=torch.bfloat16),
        W_dec=torch.zeros(1, 2, dtype=torch.bfloat16),
        b_dec=torch.zeros(2, dtype=torch.bfloat16),
    )
    host = _feature_evaluator(profile, artifact)
    host.add("a")
    decision = host.evaluate(["a"], torch.tensor([[2.0, 0.0]], dtype=torch.bfloat16), [0])[0]
    assert (decision.action, decision.params["score"]) == ("refuse", 1.0)

    with pytest.raises(DuplicateFeatureIds, match="distinct_selected_feature_ids") as refused:
        check_device_evaluator(host)
    assert isinstance(refused.value, UnsupportedDeviceBackend)
    assert refused.value.requirement == "distinct_selected_feature_ids"
    with pytest.raises(DuplicateFeatureIds):
        DeviceSharedFeatureEngine(host, 2, 2, 8, torch.device("cpu"))
    with pytest.raises(DuplicateFeatureIds):
        runtime.STATE.configure(host, [1], observation_mode="graph", evaluation_mode="device")
    runtime.STATE.reset()
    runtime.STATE.configure(host, [1], observation_mode="graph", evaluation_mode="host")

    released = load_profile(ROOT / "profiles/lmsm_transcoder.yaml")
    features = 1 + max(
        int(i) for rule in released.active_rules for i in rule.condition.config["feature_indices"]
    )
    evaluator = _feature_evaluator(released, _transcoder(features, 4))
    check_device_evaluator(evaluator)
    DeviceSharedFeatureEngine(evaluator, 2, 2, 8, torch.device("cpu"))

    sae_profile = load_profile(ROOT / "profiles/lmsm_sae.yaml")
    sae_features = 1 + max(
        int(i) for rule in sae_profile.active_rules for i in rule.condition.config["feature_indices"]
    )
    sae_evaluator = _feature_evaluator(sae_profile, _jumprelu_sae(4, sae_features))
    check_device_evaluator(sae_evaluator)
    DeviceSharedFeatureEngine(sae_evaluator, 2, 2, 8, torch.device("cpu"))


def test_device_feature_shared_by_two_rules_matches_the_host_path():
    """One feature in two rules, with different thresholds, is not a duplicate.

    Each rule has its own threshold row and both paths keep the same running
    maximum per feature, so host and device decide identically.
    """

    from lmsm.integrations.vllm_device import DeviceSharedFeatureEngine, StepFlags

    width = 8
    profile = _feature_profile(
        [
            {"feature_indices": [0, 1], "feature_thresholds": [1.0, 2.5], "threshold": 0.5},
            {"feature_indices": [1, 2], "feature_thresholds": [0.5, 1.5], "threshold": 1.0},
        ]
    )
    host = _feature_evaluator(profile, _transcoder(3, width, scale=0.4))
    engine = DeviceSharedFeatureEngine(host, 4, 4, 64, torch.device("cpu"))
    buffer = torch.zeros((16, width), dtype=torch.bfloat16)
    epoch = torch.zeros((1,), dtype=torch.int64)
    engine.bind_capture(buffer, epoch)
    ids = ["a", "b", "c", "d"]
    for request_id in ids:
        host.add(request_id)
    engine.reset_slots(torch.arange(4))
    generator = torch.Generator().manual_seed(3)
    acted, actioned = [], set()
    for step in range(10):
        live = [i for i, request_id in enumerate(ids) if request_id not in actioned]
        if not live:
            break
        buffer[: len(live)] = (torch.randn(len(live), width, generator=generator) * 2).to(torch.bfloat16)
        epoch.fill_(step + 1)
        decisions = host.evaluate([ids[i] for i in live], buffer[: len(live)], [step] * len(live))
        engine.load(np.asarray(live), np.asarray(live), np.full(len(live), step), np.arange(len(live)), step + 1)
        assert engine.run(StepFlags())
        for position, i in enumerate(live):
            hit = decisions[position].action != "allow"
            assert hit == bool(engine.result[position, 0] > 0.5), (step, ids[i])
            if hit:
                rule_index = int(engine.result[position, 1])
                assert engine.target_ids[rule_index] == decisions[position].params["target_id"]
                assert float(engine.result[position, 2]) == decisions[position].params["score"]
                acted.append((step, ids[i], decisions[position].params["rule_id"]))
                actioned.add(ids[i])
    assert {rule for _, _, rule in acted} == {rule.rule_id for rule in profile.active_rules}
    records = engine.archive([0, 1, 2, 3], [10] * 4)
    for slot, request_id in enumerate(ids):
        assert records[slot]["feature_maxima"] == host.feature_maxima_for(request_id)


# ---------------------------------------------------------------------------
# Non-finite evidence fails closed, on the host and the device evaluation path


class _Event:
    """Stands in for a CUDA event on the CPU."""

    def record(self):
        pass

    def synchronize(self):
        pass


def _nonfinite_scenario(policy, poisoned):
    """Four requests in batch-row order.  The poisoned run corrupts b, c and d.

    * a: finite evidence, and its rule fires (the same in both runs).
    * b: a NaN in its activation.
    * c: an infinite activation.  Under Anytime its score would cross the
      threshold.  Under the transcoder relu(-inf) = 0 leaves the feature
      values finite.  Under Checkpoint c finishes before the evaluation point,
      which would be not reached.
    * d: finite activations whose float32 running sum overflows (Anytime,
      Checkpoint) or a NaN feature activation (transcoder).

    Requests that finish early are the last batch rows, so their removal
    moves no other request.
    """

    nan, inf = float("nan"), float("inf")
    if policy == "anytime":
        evaluator = AnytimePolicyEvaluator(
            [_probe("chemical_biological", threshold=3.0)], _profile("lmsm_anytime")
        )
        order, lengths, fires = "bcda", {"a": 7, "b": 7, "c": 7, "d": 7}, ("a", 4)
        poison = {
            ("b", 2): [nan, 0.1],
            ("c", 3): [inf, 0.1],
            ("d", 1): [-3e38, 0.0],
            ("d", 2): [-3e38, 0.0],
        }
        failures = {"b": 2, "c": 3, "d": 2}

        def clean(request_id, step):
            if request_id == "a":
                return [5.0, 5.0] if step == 4 else [0.5, 0.5]
            return [0.1, 0.1]

    elif policy == "checkpoint":
        rule = CheckpointRule(
            target_id="chemical_biological",
            prompt=_probe("chemical_biological"),
            steps_1_32=_probe("chemical_biological"),
            steps_33_64=_probe("chemical_biological"),
            fusion=_probe("chemical_biological", width=3, threshold=3.5),
        )
        evaluator = CheckpointPolicyEvaluator([rule], _profile("lmsm_checkpoint"))
        order, lengths, fires = "bdac", {"a": 70, "b": 70, "c": 51, "d": 70}, ("a", 64)
        poison = {
            ("b", 0): [nan, 1.0],
            ("c", 40): [inf, 0.5],
            ("d", 1): [-3e38, 0.0],
            ("d", 2): [-3e38, 0.0],
        }
        failures = {"b": 0, "c": 40, "d": 32}  # d: the window-1 mean at step 32

        def clean(request_id, step):
            if request_id == "a":
                return [1.0, 1.0] if step == 0 else [0.5, 0.5]
            return [0.1, 0.1]

    else:
        from lmsm.loaders.sae_loader import LinearTranscoderAdapter

        profile = _feature_profile(
            [{"feature_indices": [0, 1], "feature_thresholds": [1.0, 1.0], "threshold": 0.5}],
            hidden_width=2,
        )
        artifact = LinearTranscoderAdapter(
            W_enc=torch.tensor([[1.0, 0.5], [1.0, 1.0]]),
            b_enc=torch.zeros(2),
            W_dec=torch.zeros(1, 2),
            b_dec=torch.zeros(2),
        )
        evaluator = _feature_evaluator(profile, artifact)
        order, lengths, fires = "bcda", {"a": 7, "b": 7, "c": 7, "d": 7}, ("a", 2)
        poison = {("b", 1): [nan, 0.1], ("c", 1): [-inf, 0.0], ("d", 3): [0.1, nan]}
        failures = {"b": 1, "c": 1, "d": 3}

        def clean(request_id, step):
            return [2.0, 0.0] if (request_id, step) == ("a", 2) else [0.1, 0.1]

    def values(request_id, step):
        if poisoned and (request_id, step) in poison:
            return poison[(request_id, step)]
        return clean(request_id, step)

    return SimpleNamespace(
        evaluator=evaluator,
        order=list(order),
        lengths=lengths,
        fires=fires,
        values=values,
        failures=failures,
    )


def _run_nonfinite_scenario(policy, evaluation, *, poisoned, triton=False, device="cpu", graphs=False):
    """Run the scenario through the compiled-mode runtime and return release views.

    The graph-capture op is simulated: each forward writes the packed rows to
    the capture buffer and stamps the epoch.  Host evaluation then reads the
    buffer (``consume_graph_capture``) as under compiled vLLM.  Device
    evaluation runs the device engine (``after_forward``, ``apply_device``).
    """

    from lmsm.integrations.vllm_graph import CAPTURE

    device = torch.device(device)
    scenario = _nonfinite_scenario(policy, poisoned)
    state = runtime.STATE
    state.configure(
        scenario.evaluator, [91, 92], observation_mode="graph", evaluation_mode=evaluation
    )
    state.eos_token_id = 3
    CAPTURE.reset()
    CAPTURE.allocate(32, 2, torch.float32, device)
    CAPTURE.captured_epoch = torch.zeros((1,), dtype=torch.int64, device=device)
    rows = len(scenario.order)
    if evaluation == "device":
        state.init_device(max_rows=rows, max_steps=128, device=device)
        if device.type != "cuda":
            state.device_event = _Event()
        state.device_engine.use_graphs, state.device_engine.use_triton = graphs, triton
        state.runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                num_tokens_no_spec=np.zeros(rows, dtype=np.int64),
                num_prompt_tokens=np.zeros(rows, dtype=np.int64),
                num_computed_tokens_cpu=np.zeros(rows, dtype=np.int64),
            )
        )
    prompt = [11, 12]
    outs = {request_id: [] for request_id in scenario.order}
    live = list(scenario.order)
    state.update_batch(
        _update(
            batch_size=rows,
            added=[(row, _params(r), prompt, outs[r]) for row, r in enumerate(live)],
        )
    )
    while live:
        counts = [1 if outs[r] else len(prompt) for r in live]
        if evaluation == "device":
            _sync_batch(state, {row: (len(prompt), outs[r]) for row, r in enumerate(live)})
        state.begin_forward(counts)
        CAPTURE.buffer.zero_()
        for end, request_id in zip(np.cumsum(counts) - 1, live, strict=True):
            CAPTURE.buffer[end] = torch.tensor(scenario.values(request_id, len(outs[request_id])))
        CAPTURE.captured_epoch.copy_(CAPTURE.step_epoch)
        logits = torch.zeros(len(live), 8, device=device)
        if evaluation == "device":
            state.after_forward()
            logits = state.apply_device(logits)
        else:
            state.consume_graph_capture()
            logits = state.apply_actions(logits)
        logits = logits.cpu()
        for row, request_id in enumerate(live):
            acted = bool(torch.isinf(logits[row]).any())
            done = acted or len(outs[request_id]) + 1 == scenario.lengths[request_id]
            outs[request_id].append(3 if done else 5)
        finished = [row for row, r in enumerate(live) if outs[r][-1] == 3]
        if finished:
            assert finished == list(range(len(live) - len(finished), len(live)))
            live = live[: len(live) - len(finished)]
            state.update_batch(_update(batch_size=len(live), removed=finished))
    views = {r: runtime.buffered_output(r, list(outs[r])) for r in scenario.order}
    return scenario, views, runtime.runtime_counters()


def _nonfinite_failure_steps(record):
    return [f["step"] for f in record["runtime_failures"] if "non-finite" in f["reason"]]


def _check_nonfinite_scenario(policy, evaluation, triton=False, device="cpu", graphs=False):
    """Poisoned requests fail closed at their step, and the others release the same views."""

    _, baseline, baseline_counters = _run_nonfinite_scenario(
        policy, evaluation, poisoned=False, triton=triton, device=device, graphs=graphs
    )
    scenario, views, counters = _run_nonfinite_scenario(
        policy, evaluation, poisoned=True, triton=triton, device=device, graphs=graphs
    )
    case = (policy, evaluation, triton, device)
    fired, step = scenario.fires
    record = views[fired]["execution_record"]
    assert (record["state"], record["decision"], record["decision_step"]) == (
        COMPLETED, "refuse", step
    ), case
    assert views[fired] == baseline[fired], case  # the clean request is untouched
    for request_id, step in scenario.failures.items():
        view = views[request_id]
        record = view["execution_record"]
        assert record["state"] == EXECUTION_FAILED, (case, request_id)
        assert _nonfinite_failure_steps(record) == [step], (case, request_id)
        assert view["released_output_tokens"] == [91, 92], (case, request_id)
        assert view["action_record"] is None, (case, request_id)  # no action on bad evidence
        clean = baseline[request_id]["execution_record"]
        assert clean["state"] != EXECUTION_FAILED, (case, request_id)
        assert _nonfinite_failure_steps(clean) == [], (case, request_id)
    assert (counters["nonfinite_failures"], baseline_counters["nonfinite_failures"]) == (3, 0)
    runtime.STATE.reset()
    return views


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # numpy: overflow, inf - inf
@pytest.mark.parametrize("evaluation", ["host", "device"])
@pytest.mark.parametrize("policy", ["anytime", "checkpoint", "transcoder"])
def test_nonfinite_evidence_fails_only_that_request(policy, evaluation):
    """NaN or infinite evidence or scores fail that request closed, on both paths.

    The request gets execution_failed with the non-finite step recorded, the
    fixed refusal, and no action, even when its infinite score crossed the
    threshold or it finished before the evaluation point.  The other requests
    in the batch release exactly what they release without the corruption.
    The device path runs the PyTorch form of its step functions here.  The
    next test checks the Triton kernels.
    """

    _check_nonfinite_scenario(policy, evaluation)


def _check_nonfinite_kernels():
    for policy in ("anytime", "checkpoint"):
        _check_nonfinite_scenario(policy, "device", triton=True)


def test_nonfinite_evidence_fails_closed_in_the_triton_kernels():
    """The same scenarios with the fused Triton kernels (Triton CPU interpreter)."""

    _in_triton_interpreter("_check_nonfinite_kernels")


# ---------------------------------------------------------------------------
# Asynchronous scheduling: evidence seen only on a discarded forward


def _discarded_forward_state(async_scheduling):
    state = _device_state("anytime")
    state.async_scheduling = async_scheduling
    outs = []
    state.update_batch(_update(batch_size=1, added=[(0, _params("a"), [1, 2], outs)]))
    outs.extend([5, 6, 7, 8])  # four tokens: steps 0..3 consumed
    return state, outs


@pytest.mark.parametrize(
    "async_scheduling, nonfinite_step, failed",
    [(True, 4, False), (True, 3, True), (False, 4, True)],
)
def test_device_nonfinite_evidence_honors_the_discarded_forward_boundary(
    async_scheduling, nonfinite_step, failed
):
    """Non-finite evidence follows the same discarded-forward boundary as decisions.

    Under async scheduling vLLM can run one more forward for a request that
    already finished and discard its sampled token.  That forward is step 4
    for a four-token completion.  A non-finite value seen only there is a
    diagnostic, not a failure.  One at a consumed step (3) still fails the
    request with the fixed refusal.  Without async scheduling no forward is
    discarded, so a recorded step at or past the final length fails closed,
    as a decision there would.
    """

    state, outs = _discarded_forward_state(async_scheduling)
    state._store_archive("a", {
        "observed": np.array([0, 1, 1, 1, 1], dtype=np.uint8),
        "policy_scores": {}, "policy_score_steps": {}, "policy_actions": {},
        "evaluation_error": None, "nonfinite_step": nonfinite_step,
    })
    view = runtime.buffered_output("a", list(outs))
    record = view["execution_record"]
    counters = runtime.runtime_counters()
    if failed:
        assert record["state"] == EXECUTION_FAILED
        assert _nonfinite_failure_steps(record) == [nonfinite_step]
        assert view["released_output_tokens"] == [91, 92]
        assert "discarded_nonfinite_step" not in record
        assert (counters["nonfinite_failures"], counters["discarded_async_nonfinite"]) == (1, 0)
    else:
        assert record["state"] == COMPLETED and record["runtime_failures"] == []
        assert record["discarded_nonfinite_step"] == 4
        assert view["released_output_tokens"] == outs
        assert (counters["nonfinite_failures"], counters["discarded_async_nonfinite"]) == (0, 1)


def _async_scenario(policy, poisoned):
    """Five requests under async scheduling.  The poisoned run corrupts b and c.

    Every request runs one more forward after its final token, as under
    vLLM's async scheduling.  That forward's step is the final length, and its
    sampled token is discarded.

    * a: its rule fires at a consumed step.  The request is actioned, so the
      discarded forward is not evaluated.
    * b: a NaN (for the transcoder a -inf, hidden by relu) only on its
      discarded forward.  It is still allowed, with the diagnostic.
    * c: a non-finite value at a consumed step.  It fails closed there.
    * d: finite evidence throughout.  It is allowed.
    * e: its rule fires only on the discarded forward (both runs).  The
      decision is discarded and the request is allowed.

    Requests that leave together are the last batch rows.
    """

    base = _nonfinite_scenario(policy, poisoned=False)
    if policy == "anytime":
        order, lengths = "cdbae", {"a": 7, "b": 6, "c": 7, "d": 7, "e": 5}
        poison = {("b", 6): [float("nan"), 0.1], ("c", 3): [float("inf"), 0.1]}
        expected = {"a": 4, "b": 6, "c": 3}

        def clean(request_id, step):
            if request_id == "e":
                return [50.0, 50.0] if step == 5 else [0.1, 0.1]
            return base.values(request_id, step)

    elif policy == "checkpoint":
        order, lengths = "daecb", {"a": 70, "b": 50, "c": 51, "d": 70, "e": 64}
        poison = {("b", 50): [float("nan"), 1.0], ("c", 40): [float("inf"), 0.5]}
        expected = {"a": 64, "b": 50, "c": 40}

        def clean(request_id, step):
            return base.values("a" if request_id == "e" else request_id, step)

    else:
        order, lengths = "cdbea", {"a": 7, "b": 6, "c": 7, "d": 7, "e": 5}
        poison = {("b", 6): [float("-inf"), 0.0], ("c", 1): [float("nan"), 0.1]}
        expected = {"a": 2, "b": 6, "c": 1}

        def clean(request_id, step):
            if request_id == "e":
                return [2.0, 0.0] if step == 5 else [0.1, 0.1]
            return base.values(request_id, step)

    def values(request_id, step):
        if poisoned and (request_id, step) in poison:
            return poison[(request_id, step)]
        return clean(request_id, step)

    return SimpleNamespace(
        evaluator=base.evaluator, order=list(order), lengths=lengths, values=values,
        fires=expected["a"], discarded=expected["b"], failure=expected["c"],
    )


def _run_async_scenario(policy, *, poisoned, triton=False, device="cpu", graphs=False):
    """The device runtime under simulated async scheduling.  Returns release views."""

    from lmsm.integrations.vllm_graph import CAPTURE

    device = torch.device(device)
    scenario = _async_scenario(policy, poisoned)
    state = runtime.STATE
    state.configure(scenario.evaluator, [91, 92], observation_mode="graph", evaluation_mode="device")
    state.eos_token_id = 3
    CAPTURE.reset()
    CAPTURE.allocate(32, 2, torch.float32, device)
    CAPTURE.captured_epoch = torch.zeros((1,), dtype=torch.int64, device=device)
    rows = len(scenario.order)
    state.init_device(max_rows=rows, max_steps=128, device=device)
    state.async_scheduling = True
    if device.type != "cuda":
        state.device_event = _Event()
    state.device_engine.use_graphs, state.device_engine.use_triton = graphs, triton
    state.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_tokens_no_spec=np.zeros(rows, dtype=np.int64),
            num_prompt_tokens=np.zeros(rows, dtype=np.int64),
            num_computed_tokens_cpu=np.zeros(rows, dtype=np.int64),
        )
    )
    prompt = [11, 12]
    outs = {request_id: [] for request_id in scenario.order}
    live = list(scenario.order)
    state.update_batch(
        _update(batch_size=rows, added=[(row, _params(r), prompt, outs[r]) for row, r in enumerate(live)])
    )
    finishing: set[str] = set()  # finished: this forward is their discarded one
    while live:
        counts = [1 if outs[r] else len(prompt) for r in live]
        _sync_batch(state, {row: (len(prompt), outs[r]) for row, r in enumerate(live)})
        state.begin_forward(counts)
        CAPTURE.buffer.zero_()
        for end, request_id in zip(np.cumsum(counts) - 1, live, strict=True):
            CAPTURE.buffer[end] = torch.tensor(scenario.values(request_id, len(outs[request_id])))
        CAPTURE.captured_epoch.copy_(CAPTURE.step_epoch)
        state.after_forward()
        logits = state.apply_device(torch.zeros(len(live), 8, device=device)).cpu()
        done_now = set()
        for row, request_id in enumerate(live):
            if request_id in finishing:
                continue  # vLLM discards the token sampled in this forward
            acted = bool(torch.isinf(logits[row]).any())
            done = acted or len(outs[request_id]) + 1 == scenario.lengths[request_id]
            outs[request_id].append(3 if done else 5)
            if done:
                done_now.add(request_id)
        leaving = [row for row, r in enumerate(live) if r in finishing]
        if leaving:
            assert leaving == list(range(len(live) - len(leaving), len(live)))
            live = live[: len(live) - len(leaving)]
            state.update_batch(_update(batch_size=len(live), removed=leaving))
        finishing = done_now
    views = {r: runtime.buffered_output(r, list(outs[r])) for r in scenario.order}
    return scenario, views, runtime.runtime_counters()


def _check_async_scenario(policy, triton=False, device="cpu", graphs=False):
    """Only the consumed steps' evidence decides, and the other requests release the same views."""

    _, baseline, baseline_counters = _run_async_scenario(
        policy, poisoned=False, triton=triton, device=device, graphs=graphs
    )
    scenario, views, counters = _run_async_scenario(
        policy, poisoned=True, triton=triton, device=device, graphs=graphs
    )
    case = (policy, triton, device)
    for request_id in "ade":
        assert views[request_id] == baseline[request_id], (case, request_id)
    record = views["a"]["execution_record"]
    assert (record["state"], record["decision"], record["decision_step"]) == (
        COMPLETED, "refuse", scenario.fires
    ), case
    assert views["e"]["action_record"] is None, case  # decided only on its discarded forward
    assert views["e"]["released_output_tokens"] == views["e"]["ordinary_model_prefix_tokens"], case

    record = views["b"]["execution_record"]
    assert record["discarded_nonfinite_step"] == scenario.discarded, case
    assert record["runtime_failures"] == [] and record["state"] != EXECUTION_FAILED, case
    clean = baseline["b"]
    assert "discarded_nonfinite_step" not in clean["execution_record"], case
    stripped = {k: v for k, v in record.items() if k != "discarded_nonfinite_step"}
    assert dict(views["b"], execution_record=stripped) == clean, case

    view = views["c"]
    record = view["execution_record"]
    assert record["state"] == EXECUTION_FAILED, case
    assert _nonfinite_failure_steps(record) == [scenario.failure], case
    assert view["released_output_tokens"] == [91, 92] and view["action_record"] is None, case
    assert baseline["c"]["execution_record"]["state"] != EXECUTION_FAILED, case

    def picked(values):
        keys = ("nonfinite_failures", "discarded_async_nonfinite", "discarded_async_decisions")
        return tuple(values[key] for key in keys)

    assert picked(counters) == (1, 1, 1), case
    assert picked(baseline_counters) == (0, 0, 1), case
    runtime.STATE.reset()
    return views


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("policy", ["anytime", "checkpoint", "transcoder"])
def test_async_discarded_forward_evidence_does_not_fail_and_others_are_isolated(policy):
    """Mixed clean and corrupted requests under async scheduling (device path).

    A non-finite value only on a request's discarded forward leaves it
    allowed, with ``discarded_nonfinite_step``.  One at a consumed step still
    fails that request.  A decision made only on a discarded forward is
    dropped.  Every other request releases exactly what it releases without
    the corruption.
    """

    _check_async_scenario(policy)


def _check_async_kernels():
    for policy in ("anytime", "checkpoint", "transcoder"):
        _check_async_scenario(policy, triton=True)


def test_async_discarded_forward_evidence_in_the_triton_kernels():
    """The same scenarios with the fused kernels (Triton CPU interpreter)."""

    _in_triton_interpreter("_check_async_kernels")


# ---------------------------------------------------------------------------
# The fused finiteness checks against their PyTorch references


def _same_values(a, b):
    """Bitwise equal, NaN in the same places (NaN != NaN for torch.equal)."""

    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if not a.is_floating_point():
        return torch.equal(a, b)
    keep = dict(nan=0.0, posinf=float("inf"), neginf=float("-inf"))
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(**keep), b.nan_to_num(**keep))


DENSE_NONFINITE_CASES = [
    # policy, width, LMSM_ANYTIME_BLOCK_D, zero the poisoned column's probe weights
    ["anytime", 40, 16, False],
    ["anytime", 40, 16, True],  # inf * 0: the scores still catch it
    ["anytime", 64, None, True],
    ["checkpoint", 300, None, False],
]


def _check_dense_nonfinite_agreement(cases, device="cpu"):
    """Kernel engines against the PyTorch reference with NaN, +-inf and overflow.

    The Anytime kernel checks only the probe scores, since a non-finite pooled
    value makes every score non-finite.  The reference checks the pooled mean
    and the scores.  They must record the same first steps and act
    identically, including when the poisoned dimension has zero weight in
    every probe (inf * 0 = NaN).
    """

    import os

    graphs = torch.device(device).type == "cuda"
    for policy, width, block, zero_column in cases:
        if block is None:
            os.environ.pop("LMSM_ANYTIME_BLOCK_D", None)
        else:
            os.environ["LMSM_ANYTIME_BLOCK_D"] = str(block)
        evaluator = _dense_evaluator(policy, width)
        kernel = _dense_engine(evaluator, triton=True, device=device, graphs=graphs)
        reference = _dense_engine(evaluator, triton=False, device=device, graphs=graphs)
        if zero_column:
            for engine in (kernel, reference):
                engine.weight[width // 2].zero_()
        steps = _policy_steps(kernel)
        last = steps[-1]
        nan, inf = float("nan"), float("inf")
        # Rows 0..2 are slots 0..2, and slot 3 is padding.  Row 2 overflows in
        # float32 from finite values (two near-maximal activations).
        poison = {(0, 2): nan, (1, 3): inf, (1, 5): -inf, (2, 4): 3.0e38, (2, 5): 3.0e38}
        if policy == "checkpoint":
            # Row 2 overflows window 1, as its mean at step 32 shows.  Slot 3 is
            # padding here and is not involved.  Row 0 overflows window 2 later,
            # after its NaN prompt, and the first step (0) is kept.
            poison = {
                (0, 0): nan, (1, 20): inf, (2, 10): 3.0e38, (2, 11): 3.0e38,
                (0, 40): 3.0e38, (0, 41): 3.0e38,
            }
        buffers = [torch.zeros((16, width), dtype=torch.bfloat16, device=device) for _ in range(2)]
        traces = _run_dense_steps([kernel, reference], buffers, width, steps, poison=poison)
        case = (policy, width, block, zero_column)
        names = ("observed", "nonfinite", "sums", "counts", "max_steps_tensor", "crossed")
        if policy == "checkpoint":
            names = ("observed", "nonfinite", "window_sums", "window_counts", "done", "crossed", "errors")
        for name in names:
            assert _same_values(getattr(kernel, name), getattr(reference, name)), (name, case)
        for (k_produced, k_result, k_act, k_ok), (r_produced, r_result, r_act, r_ok) in zip(
            *traces, strict=True
        ):
            assert (k_produced, k_ok) == (r_produced, r_ok), case
            assert torch.equal(k_act, r_act) and torch.equal(k_result[:, :2], r_result[:, :2]), case
        first = kernel.nonfinite[:3].tolist()
        if policy == "anytime":
            assert first == [2, 3, 5], case  # row 2: the float32 sum overflows at step 5
        else:
            assert first == [0, 20, 32], case  # row 2: the window-1 mean at step 32
        assert last >= max(step for _, step in poison), case


def test_triton_kernels_record_the_same_nonfinite_steps_as_the_reference():
    _in_triton_interpreter("_check_dense_nonfinite_agreement", DENSE_NONFINITE_CASES)


def _feature_engine_pair(width, features, *, actions=True, seed=0, device="cpu"):
    from lmsm.aggregators.max_score import MaxScoreAggregator
    from lmsm.build import build_shared_feature_backends
    from lmsm.integrations.vllm_device import DeviceSharedFeatureEngine
    from lmsm.policies.threshold import ThresholdPolicy
    from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator

    profile = _feature_profile(
        [{"feature_indices": list(range(features)), "feature_thresholds": [1.0] * features,
          "threshold": 0.5}],
        hidden_width=width,
    )
    shared, bound = build_shared_feature_backends(
        profile, profile.backend_bindings[0], _transcoder(features, width, seed=seed), selected=True
    )
    evaluator = SharedFeaturePolicyEvaluator(
        shared, bound, MaxScoreAggregator(), ThresholdPolicy(actions_enabled=actions), profile
    )
    pair = []
    for triton in (True, False):
        engine = DeviceSharedFeatureEngine(evaluator, 6, 6, 32, torch.device(device))
        engine.use_triton = triton
        pair.append(engine)
    return pair


def _check_feature_kernel(device="cpu", widths=((40, 3), (1500, 70))):
    """The fused selected-feature check against its reference, row by row.

    The cases are widths that no block divides, and a -inf, NaN or +inf in the
    first or last activation of a row next to a clean row.  A read past a row
    would flag the clean neighbor.  Further cases are non-finite feature
    activations, a step whose capture was not delivered (no record), and a
    second non-finite step (the first is kept).
    """

    nan, inf = float("nan"), float("inf")
    for width, features in widths:
        for actions in (True, False):
            engines = _feature_engine_pair(width, features, actions=actions, device=device)
            for engine in engines:
                engine.reset_slots(torch.arange(engine.num_slots, device=engine.device))
            generator = torch.Generator().manual_seed(width)
            history = []
            for step, ok in ((1, True), (2, True), (3, False), (4, True)):
                hidden = torch.randn(5, width, generator=generator).to(torch.bfloat16)
                scores = torch.rand(5, features, generator=generator, dtype=torch.float64)
                if step == 2:
                    hidden[0, width - 1] = nan  # row 1 starts right after it
                    hidden[2, 0] = -inf  # right after row 1's last value
                    scores[3, features - 1] = inf
                if step == 3:  # not delivered: nothing is recorded
                    hidden[1, 0] = nan
                if step == 4:
                    hidden[0, 0] = inf  # already recorded at step 2: kept
                    scores[4, 0] = nan
                slots = np.array([4, 2, 0, 1, 3])
                allowed = []
                for engine in engines:
                    engine.load(np.arange(5), slots, np.full(5, step), np.arange(5), 0)
                    engine.ok.fill_(ok)
                    index = engine.index
                    result = engine._check_evidence(
                        index[1, :5], index[2, :5], hidden.to(device), scores.to(device)
                    )
                    assert (result is None) == (not actions)
                    allowed.append(None if result is None else result.clone())
                assert torch.equal(engines[0].nonfinite, engines[1].nonfinite), (width, step)
                if actions:
                    assert torch.equal(allowed[0], allowed[1]), (width, step)
                history.append(engines[0].nonfinite[slots].tolist())
            assert history[-1] == [2, -1, 2, 2, 4], (width, actions, history)
            if actions:
                assert allowed[0].tolist() == [False, True, False, False, False]


def _check_feature_engine_kernel(device="cpu", widths=((40, 3), (1500, 70))):
    """Whole steps: the engine with the kernel equals the engine with the reference."""

    from lmsm.integrations.vllm_device import StepFlags

    nan, inf = float("nan"), float("inf")
    for width, features in widths:
        engines = _feature_engine_pair(width, features, device=device)
        buffer = torch.zeros((16, width), dtype=torch.bfloat16, device=device)
        epoch = torch.zeros((1,), dtype=torch.int64, device=device)
        for engine in engines:
            engine.bind_capture(buffer, epoch)
            engine.reset_slots(torch.arange(engine.num_slots, device=engine.device))
        generator = torch.Generator().manual_seed(7)
        for step in range(6):
            buffer[:5] = (torch.randn(5, width, generator=generator) * 2).to(torch.bfloat16)
            if step == 2:
                buffer[1, width // 2] = -inf  # relu hides it from the features
                buffer[3, 0] = nan
            if step == 4:
                # Finite activations whose bf16 feature 0 overflows to +inf.
                sign = torch.sign(engines[0].encoder.weight[0].float())
                buffer[4, :] = (sign * 3.0e38).to(torch.bfloat16).to(device)
            epoch.fill_(step + 1)
            results = []
            for engine in engines:
                engine.load(np.arange(5), np.arange(5), np.full(5, step), np.arange(5), step + 1)
                produced = engine.run(StepFlags())
                results.append((produced, engine.result[:5].clone(), engine.act_rows.clone()))
            assert results[0][0] == results[1][0]
            assert _same_values(results[0][1], results[1][1]), (width, step)
            assert torch.equal(results[0][2], results[1][2]), (width, step)
        for name in ("observed", "nonfinite", "maxima", "evaluated"):
            assert _same_values(getattr(engines[0], name), getattr(engines[1], name)), (width, name)
        assert engines[0].nonfinite[:5].tolist() == [-1, 2, -1, 2, 4], width


def test_selected_feature_kernel_matches_its_reference():
    """The fused selected-feature check (Triton CPU interpreter) equals the PyTorch form."""

    _in_triton_interpreter("_check_feature_kernel")
    _in_triton_interpreter("_check_feature_engine_kernel")

