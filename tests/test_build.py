from pathlib import Path

import pytest

from lmsm.build import build_policy_evaluator
from lmsm.loaders.profile_loader import load_profile
from lmsm.policy_evaluator import (
    AnytimePolicyEvaluator,
    CheckpointPolicyEvaluator,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "evaluator_type", "attribute", "count"),
    (
        (
            "lmsm_anytime.yaml",
            AnytimePolicyEvaluator,
            "probes",
            15,
        ),
        (
            "lmsm_checkpoint.yaml",
            CheckpointPolicyEvaluator,
            "rules",
            15,
        ),
    ),
)
def test_build_policy_evaluator_preserves_active_target_order(
    filename, evaluator_type, attribute, count
):
    profile = load_profile(ROOT / "profiles" / filename)
    evaluator = build_policy_evaluator(profile, ROOT)
    loaded = getattr(evaluator, attribute)
    loaded_targets = tuple(
        getattr(item, "target_id", getattr(item, "category", None)) for item in loaded
    )

    assert isinstance(evaluator, evaluator_type)
    assert len(loaded) == count
    assert loaded_targets == profile.enabled_targets
