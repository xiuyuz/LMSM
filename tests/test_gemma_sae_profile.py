from pathlib import Path

import torch

from lmsm.aggregators.max_score import MaxScoreAggregator
from lmsm.build import build_shared_feature_backends
from lmsm.loaders.profile_loader import load_profile
from lmsm.policies.threshold import ThresholdPolicy
from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "profiles/lmsm_sae.yaml"

EXPECTED_RULES = {
    "chemical_biological": (
        [6681, 13810, 24218, 7074],
        [450.0650329589844, 233.44338989257812, 303.9464416503906, 1458.8687744140625],
        0.75,
    ),
    "cybercrime_intrusion": (
        [51489, 68223],
        [301.3700866699219, 231.80169677734375],
        0.5,
    ),
    "illegal_activity": (
        [542, 4964, 20395],
        [473.61322021484375, 1082.739990234375, 314.9058837890625],
        0.6666666666666666,
    ),
    "misinformation_disinformation": (
        [8633, 11056, 212973, 112822],
        [582.9880981445312, 611.2312622070312, 268.3076171875, 318.1142272949219],
        0.5,
    ),
    "book_copyright": ([222799], [620.438720703125], 1.0),
    "lyrics_copyright": ([120912], [560.1005249023438], 1.0),
}


class FullEncodeOnlySAE:
    def __init__(self):
        self.full_calls = 0

    def encode(self, hidden):
        encoded = torch.zeros(hidden.shape[0], 222800)
        feature_id = (6681, 13810, 24218)[min(self.full_calls, 2)]
        thresholds = EXPECTED_RULES["chemical_biological"][1]
        feature_position = (6681, 13810, 24218).index(feature_id)
        encoded[:, feature_id] = thresholds[feature_position] + 1.0
        self.full_calls += 1
        return encoded


def test_gemma_sae_profile_preserves_calibrated_bundle():
    profile = load_profile(PROFILE_PATH)
    binding = profile.backend_bindings[0]

    assert profile.name == profile.policy.name == "LMSM-SAE"
    assert profile.policy.active_rule_ids == tuple(EXPECTED_RULES)
    assert binding.backend_type == "sae"
    assert binding.activation_key == "model.language_model.layers.22"
    assert binding.config == {
        "model_id": "google/gemma-3-4b-it",
        "hidden_width": 2560,
    }
    assert binding.artifact == {
        "source": "hf",
        "release": "gemma-scope-2-4b-it-res",
        "sae_id": "layer_22_width_262k_l0_medium",
    }
    assert profile.policy.action_mapping.resolve("refuse") == "refuse"
    assert profile.policy.schedule.kind == "anytime"
    assert profile.policy.schedule.pooling == "cumulative_feature_maxima"
    assert profile.policy.schedule.first_crossing is True

    for rule in profile.active_rules:
        indices, feature_thresholds, policy_threshold = EXPECTED_RULES[rule.rule_id]
        config = rule.condition.config
        assert config["feature_indices"] == indices
        assert config["feature_threshold"] == 1.0
        assert config["feature_thresholds"] == feature_thresholds
        assert config["feature_group_rule"] == "cumulative_fraction"
        assert config["min_feature_fraction"] == policy_threshold
        assert config["threshold"] == policy_threshold
        assert rule.candidate_action == "refuse"


def test_gemma_sae_full_encode_fallback_reaches_cumulative_policy_threshold():
    profile = load_profile(PROFILE_PATH)
    artifact = FullEncodeOnlySAE()
    shared_backend, bound_backends = build_shared_feature_backends(
        profile,
        profile.backend_bindings[0],
        artifact,
        selected=True,
    )
    evaluator = SharedFeaturePolicyEvaluator(
        shared_backend,
        bound_backends,
        MaxScoreAggregator(),
        ThresholdPolicy(),
        profile,
    )
    evaluator.add("gemma-request")

    decisions = [
        evaluator.evaluate(
            ["gemma-request"],
            torch.zeros(1, 4),
            [step],
        )[0]
        for step in range(3)
    ]

    assert [decision.action for decision in decisions] == ["allow", "allow", "refuse"]
    assert decisions[-1].params["rule_id"] == "chemical_biological"
    assert decisions[-1].params["score"] == 0.75
    assert artifact.full_calls == 3
    assert shared_backend.encode_count == 3
    assert bound_backends[0].threshold == 1.0
