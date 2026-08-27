from pathlib import Path

import numpy as np
import torch

from lmsm.backends.dense_probe import CheckpointRule, DenseProbe
from lmsm.loaders.profile_loader import load_profile
from lmsm.policy_evaluator import AnytimePolicyEvaluator, CheckpointPolicyEvaluator


ROOT = Path(__file__).resolve().parents[1]


def _probe(target_id: str, *, width: int = 1, threshold: float = 0.5):
    return DenseProbe(
        target_id=target_id,
        mean=np.zeros(width, dtype=np.float64),
        scale=np.ones(width, dtype=np.float64),
        coef=np.ones(width, dtype=np.float64),
        intercept=np.float64(0.0),
        threshold=np.float64(threshold),
    )


def _checkpoint_rule(target_id: str):
    return CheckpointRule(
        target_id=target_id,
        prompt=_probe(target_id, threshold=0.0),
        steps_1_32=_probe(target_id, threshold=0.0),
        steps_33_64=_probe(target_id, threshold=0.0),
        fusion=_probe(target_id, width=3, threshold=2.0),
    )


def test_anytime_first_crossing_uses_fixed_or_active_rule_order():
    profile = load_profile(ROOT / "profiles/lmsm_anytime.yaml")
    profile = profile.with_active_rules(("copyright", "chemical_biological"))
    evaluator = AnytimePolicyEvaluator(
        [_probe("chemical_biological"), _probe("copyright")], profile
    )
    evaluator.add("request")

    before = evaluator.evaluate(["request"], torch.tensor([[0.0]]), [1])[0]
    crossing = evaluator.evaluate(["request"], torch.tensor([[2.0]]), [2])[0]
    after = evaluator.evaluate(["request"], torch.tensor([[2.0]]), [3])[0]

    assert before.action == "allow"
    assert crossing.action == "refuse"
    assert crossing.params["rule_id"] == "copyright"
    assert crossing.params["target_id"] == "copyright"
    assert crossing.params["score"] == 1.0
    assert crossing.params["threshold"] == 0.5
    assert after.action == "allow"
    assert evaluator.actions_for("request") == {
        "chemical_biological": True,
        "copyright": True,
    }


def test_checkpoint_decides_only_at_step_64_in_fixed_or_active_rule_order():
    profile = load_profile(ROOT / "profiles/lmsm_checkpoint.yaml")
    profile = profile.with_active_rules(("copyright", "chemical_biological"))
    evaluator = CheckpointPolicyEvaluator(
        [_checkpoint_rule("chemical_biological"), _checkpoint_rule("copyright")],
        profile,
    )
    evaluator.add("request")

    decisions = [
        evaluator.evaluate(["request"], torch.ones(1, 1), [step])[0]
        for step in range(65)
    ]

    assert all(decision.action == "allow" for decision in decisions[:64])
    decision = decisions[64]
    assert decision.action == "refuse"
    assert decision.params["rule_id"] == "copyright"
    assert decision.params["target_id"] == "copyright"
    assert decision.params["threshold"] == 2.0
    assert decision.params["score"] == 3.0
    assert decision.params["prompt_score"] == 1.0
    assert decision.params["steps_1_32_score"] == 1.0
    assert decision.params["steps_33_64_score"] == 1.0
    assert evaluator.score_steps_for("request") == {
        "chemical_biological": 64,
        "copyright": 64,
    }
