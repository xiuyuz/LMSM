"""CPU tests for fixed-OR composition of the Checkpoint probe and transcoder policies."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmsm.aggregators.max_score import MaxScoreAggregator
from lmsm.build import build_policy_evaluator, build_shared_feature_backends
from lmsm.composition import ComposedPolicyEvaluator, load_composed_profile
from lmsm.loaders.profile_loader import load_profile
from lmsm.loaders.sae_loader import LinearTranscoderAdapter
from lmsm.policies.threshold import ThresholdPolicy
from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator
from lmsm.state import Decision

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"
WIDTH = 2560


def _transcoder(profile, seed=0):
    generator = torch.Generator().manual_seed(seed)
    features = 1 + max(int(i) for r in profile.active_rules for i in r.condition.config["feature_indices"])
    w_enc = torch.zeros(features, WIDTH)
    # Only the referenced features get weights (the rest never matter), scaled so that they cross sometimes.
    for rule in profile.active_rules:
        for feature in rule.condition.config["feature_indices"]:
            w_enc[int(feature)] = torch.randn(WIDTH, generator=generator) * 0.004
    return LinearTranscoderAdapter(
        W_enc=w_enc.to(torch.bfloat16),
        b_enc=torch.zeros(features, dtype=torch.bfloat16),
        W_dec=torch.zeros(1, WIDTH, dtype=torch.bfloat16),
        b_dec=torch.zeros(WIDTH, dtype=torch.bfloat16),
    )


def _member_evaluators(members, artifact):
    out = []
    for member in members:
        binding = member.profile.backend_bindings[0]
        if binding.backend_type == "transcoder":
            shared, bound = build_shared_feature_backends(member.profile, binding, artifact, selected=True)
            evaluator = SharedFeaturePolicyEvaluator(
                shared, bound, MaxScoreAggregator(), ThresholdPolicy(actions_enabled=True), member.profile
            )
        else:
            evaluator = build_policy_evaluator(member.profile, ROOT, actions_enabled=True, device="cpu")
        out.append((member.name, evaluator))
    return out


def _plain(profile_name, artifact):
    profile = load_profile(PROFILES / profile_name)
    binding = profile.backend_bindings[0]
    if binding.backend_type == "transcoder":
        shared, bound = build_shared_feature_backends(profile, binding, artifact, selected=True)
        return SharedFeaturePolicyEvaluator(
            shared, bound, MaxScoreAggregator(), ThresholdPolicy(actions_enabled=True), profile
        )
    return build_policy_evaluator(profile, ROOT, actions_enabled=True, device="cpu")


def _drive(evaluator, n_requests=6, steps=80, seed=1, scale=3.0):
    """Feed every request its own random rows through the host-path eligibility and return the first actions."""

    generator = torch.Generator().manual_seed(seed)
    ids = [f"r{i}" for i in range(n_requests)]
    for rid in ids:
        evaluator.add(rid)
    acted = {}
    for step in range(steps + 1):
        live = [
            rid
            for rid in ids
            if rid not in acted
            and (
                (step == 0 and evaluator.observes_prompt and evaluator.needs_prompt(rid))
                or (
                    step > 0
                    and evaluator.observes_decode
                    and (evaluator.max_decode_step is None or step <= evaluator.max_decode_step)
                )
            )
        ]
        if not live:
            continue
        rows = (torch.randn(len(ids), WIDTH, generator=generator) * scale).to(torch.bfloat16)
        rows = rows[[ids.index(rid) for rid in live]]
        for rid, decision in zip(live, evaluator.evaluate(live, rows, [step] * len(live)), strict=True):
            if decision.action != "allow":
                acted[rid] = (step, decision.params["rule_id"], decision.params.get("target_id"))
    return acted


def test_composed_profiles_add_rules_in_member_order():
    profile, members = load_composed_profile(PROFILES / "lmsm_composed_all.yaml")
    assert [m.name for m in members] == ["probe", "transcoder"]
    ids = profile.policy.active_rule_ids
    assert len(ids) == 21 and ids[0] == "probe/chemical_biological" and ids[15] == "transcoder/chemical_biological"
    assert profile.policy.composition.kind == "fixed_or"
    assert {b.activation_key for b in profile.backend_bindings} == {"pre:model.layers.24.mlp"}
    shared, members = load_composed_profile(PROFILES / "lmsm_composed_shared.yaml")
    assert shared.policy.active_rule_ids[15:] == (
        "transcoder/chemical_biological",
        "transcoder/cybercrime_intrusion",
        "transcoder/misinformation_disinformation",
    )
    probe_only, members = load_composed_profile(PROFILES / "lmsm_composed_check_probe.yaml")
    assert [m.name for m in members] == ["probe"] and len(probe_only.policy.active_rule_ids) == 15


def test_union_requirements_are_exact():
    artifact = _transcoder(load_profile(PROFILES / "lmsm_transcoder.yaml"))
    plain_probe = _plain("lmsm_checkpoint.yaml", artifact).execution_requirements()
    plain_transcoder = _plain("lmsm_transcoder.yaml", artifact).execution_requirements()
    for name, plain in (("lmsm_composed_check_probe.yaml", plain_probe), ("lmsm_composed_check_transcoder.yaml", plain_transcoder)):
        profile, members = load_composed_profile(PROFILES / name)
        composed = ComposedPolicyEvaluator(_member_evaluators(members, artifact), profile).execution_requirements()
        assert composed.windows == plain.windows and composed.evaluation_steps == plain.evaluation_steps
        assert composed.not_reached_disposition == plain.not_reached_disposition
    profile, members = load_composed_profile(PROFILES / "lmsm_composed_all.yaml")
    union = ComposedPolicyEvaluator(_member_evaluators(members, artifact), profile).execution_requirements()
    assert union.evaluation_steps is None  # every observation of the running transcoder schedule
    for step in (0, 1, 32, 33, 64, 65, 1000):
        assert union.requires_observation(step, None) and union.is_evaluation_point(step, None)
    assert not union.requires_observation(70, 64)  # nothing after the decision step


@pytest.mark.parametrize(
    "composed_name, plain_name",
    [("lmsm_composed_check_probe.yaml", "lmsm_checkpoint.yaml"), ("lmsm_composed_check_transcoder.yaml", "lmsm_transcoder.yaml")],
)
def test_one_active_member_reproduces_the_plain_policy(composed_name, plain_name):
    artifact = _transcoder(load_profile(PROFILES / "lmsm_transcoder.yaml"))
    for seed, scale in ((1, 3.0), (2, 6.0), (3, 12.0)):
        plain = _drive(_plain(plain_name, artifact), seed=seed, scale=scale)
        profile, members = load_composed_profile(PROFILES / composed_name)
        composed = _drive(ComposedPolicyEvaluator(_member_evaluators(members, artifact), profile), seed=seed, scale=scale)
        assert {k: (s, r.split("/", 1)[1], t) for k, (s, r, t) in composed.items()} == plain


class _Stub:
    """A member that fires its rule at fixed steps (tests the OR and the tie-break)."""

    observes_prompt = True
    observes_decode = True

    def __init__(self, rule_id, fire_at, max_decode_step=None):
        self.rule_id, self.fire_at, self.max_decode_step = rule_id, fire_at, max_decode_step
        self.seen = []

    def add(self, request_id):
        pass

    def remove(self, request_id):
        pass

    def needs_prompt(self, request_id):
        return True

    def evaluate(self, request_ids, activations, steps):
        self.seen.extend(steps)
        return [
            Decision("refuse", {"rule_id": self.rule_id, "target_id": self.rule_id, "score": 1.0})
            if step in self.fire_at
            else Decision("allow")
            for step in steps
        ]

    def scores_for(self, request_id):
        return {}

    def feature_maxima_for(self, request_id):
        return {}


def test_or_composition_takes_the_first_firing_rule_and_breaks_ties_by_rule_order():
    profile = SimpleNamespace(policy=SimpleNamespace(active_rule_ids=("probe/a", "transcoder/b")))
    probe = _Stub("a", {64}, max_decode_step=64)
    transcoder = _Stub("b", {64, 70})
    composed = ComposedPolicyEvaluator([("probe", probe), ("transcoder", transcoder)], profile)
    assert composed.max_decode_step is None
    composed.add("x")
    rows = torch.zeros(1, 4)
    assert composed.evaluate(["x"], rows, [63])[0].action == "allow"
    decision = composed.evaluate(["x"], rows, [64])[0]
    assert decision.params["rule_id"] == "probe/a" and decision.params["co_fired"] == ["transcoder/b"]
    assert decision.params["backend"] == "probe"
    composed.evaluate(["x"], rows, [70])
    assert 70 not in probe.seen and 70 in transcoder.seen  # each member sees only its own schedule's steps
    assert composed.score_steps_for("x") == {"probe/a": 64, "transcoder/b": 64}
    # Observe-only: no action, but each member's first firing is still recorded (analysis only).
    quiet = ComposedPolicyEvaluator(
        [("probe", _Stub("a", set(), 64)), ("transcoder", _Stub("b", {5}))], profile, actions_enabled=False
    )
    quiet.add("y")
    assert quiet.evaluate(["y"], rows, [5])[0].action == "allow"
    assert quiet.actions_for("y") == {"transcoder/b": True}


def test_composed_evaluator_reports_the_first_nonfinite_step_of_its_members():
    """The runtime fails a composed request closed when any member met non-finite values."""

    from lmsm.backends.dense_probe import DenseProbe
    from lmsm.policy_evaluator import AnytimePolicyEvaluator

    anytime = AnytimePolicyEvaluator(
        [DenseProbe("chemical_biological", np.zeros(2), np.ones(2), np.ones(2), np.float64(0.0), np.float64(1e9))],
        load_profile(PROFILES / "lmsm_anytime.yaml"),
    )
    profile = SimpleNamespace(policy=SimpleNamespace(active_rule_ids=("probe/a", "stub/b")))
    composed = ComposedPolicyEvaluator([("probe", anytime), ("stub", _Stub("b", set()))], profile)
    for request_id in ("x", "y"):
        composed.add(request_id)
    for step in (1, 2, 3):
        value = float("nan") if step == 2 else 0.5
        decisions = composed.evaluate(["x", "y"], torch.tensor([[value, 0.5], [0.5, 0.5]]), [step, step])
        assert [d.action for d in decisions] == ["allow", "allow"]
    assert (composed.nonfinite_step_for("x"), composed.nonfinite_step_for("y")) == (2, None)
    composed.remove("x")
    assert composed.nonfinite_step_for("x") is None
