from pathlib import Path

import pytest

from lmsm.control_plane import Composition, Schedule
from lmsm.loaders.profile_loader import load_profile


ROOT = Path(__file__).resolve().parents[1]

FIFTEEN_TARGETS = (
    "chemical_biological",
    "cybercrime_intrusion",
    "drugs_controlled_substances",
    "copyright",
    "misinformation_disinformation",
    "defamation",
    "hate_discrimination",
    "harassment_bullying",
    "fraud_deception",
    "illegal_goods_services",
    "violence_physical_harm",
    "sexual_abuse_exploitation",
    "self_harm",
    "privacy_sensitive_information",
    "general_dangerous_harm",
)

@pytest.mark.parametrize(
    (
        "filename",
        "name",
        "targets",
        "schedule",
        "action",
        "report_path",
    ),
    (
        (
            "lmsm_checkpoint.yaml",
            "LMSM-Checkpoint",
            FIFTEEN_TARGETS,
            Schedule(
                kind="checkpoint",
                pooling="fixed_window_mean",
                evaluation_step=64,
                window_split_step=32,
                max_decode_step=64,
            ),
            "refuse",
            "profiles/parameters/lmsm_checkpoint.json",
        ),
        (
            "lmsm_anytime.yaml",
            "LMSM-Anytime",
            FIFTEEN_TARGETS,
            Schedule(
                kind="anytime",
                pooling="running_prefix_mean",
                first_crossing=True,
            ),
            "refuse",
            "profiles/parameters/lmsm_anytime.json",
        ),
    ),
)
def test_release_profiles_load_exact_policy_contract(
    filename, name, targets, schedule, action, report_path
):
    profile = load_profile(ROOT / "profiles" / filename)

    assert profile.name == name
    assert profile.policy.name == name
    assert profile.policy.version == "1"
    assert profile.rule_library.version == "1"
    assert profile.backend_bindings[0].version == "1"
    assert {rule.version for rule in profile.rule_library.rules} == {"1"}
    assert tuple(target.target_id for target in profile.targets) == targets
    assert profile.enabled_targets == targets
    assert profile.policy.active_rule_ids == targets
    assert profile.policy.schedule == schedule
    assert profile.policy.composition == Composition(
        kind="fixed_or", tie_break="active_rule_order"
    )
    assert [profile.action_for_rule(rule.rule_id) for rule in profile.active_rules] == [
        action
    ] * len(targets)
    assert profile.backend_bindings[0].report_path == report_path
