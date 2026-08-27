from pathlib import Path

import torch

from lmsm.aggregators.max_score import MaxScoreAggregator
from lmsm.build import build_shared_feature_backends
from lmsm.loaders.profile_loader import load_profile
from lmsm.policies.threshold import ThresholdPolicy
from lmsm.policy_evaluator import SharedFeaturePolicyEvaluator


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "profiles/lmsm_transcoder.yaml"

EXPECTED_RULES = {
    "chemical_biological": (
        [25287, 23449, 39241],
        [0.490234375, 0.1162109375, 7.5625],
        0.3333333333333333,
        0.3333333333333333,
    ),
    "cybercrime_intrusion": ([44293], [2.21875], 1.0, 1.0),
    "illegal_activity": (
        [29939, 15780, 146551, 14615, 25868],
        [0.37890625, 3.5625, 12.875, 5.90625, 2.078125],
        0.6,
        0.6,
    ),
    "misinformation_disinformation": (
        [88583, 49816, 28072, 62341],
        [1.3359375, 5.0625, 3.59375, 0.23828125],
        0.5,
        0.5,
    ),
    "book_copyright": ([66477], [18.0], 1.0, 1.0),
    "lyrics_copyright": ([5332], [0.796875], 1.0, 1.0),
}


class FeatureTableTranscoder:
    def __init__(self, columns):
        self.columns = columns

    def encode_selected(self, hidden, feature_indices):
        return torch.stack(
            [
                hidden[:, self.columns[index]]
                if index in self.columns
                else torch.zeros(hidden.shape[0])
                for index in feature_indices
            ],
            dim=1,
        )


def _evaluator(profile, artifact):
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
    return evaluator, bound_backends


def test_qwen_transcoder_profile_preserves_calibrated_bundle():
    profile = load_profile(PROFILE_PATH)
    binding = profile.backend_bindings[0]

    assert profile.name == profile.policy.name == "LMSM-Transcoder"
    assert profile.policy.active_rule_ids == tuple(EXPECTED_RULES)
    assert binding.backend_type == "transcoder"
    assert binding.activation_key == "pre:model.layers.24.mlp"
    assert binding.config == {
        "model_id": "Qwen/Qwen3-4B",
        "hidden_width": 2560,
    }
    assert binding.artifact == {
        "source": "hf_hub",
        "repo_id": "mwhanna/qwen3-4b-transcoders",
        "filename": "layer_24.safetensors",
    }
    assert profile.policy.schedule.kind == "anytime"
    assert profile.policy.schedule.pooling == "cumulative_feature_maxima"
    assert profile.policy.schedule.first_crossing is True

    for rule in profile.active_rules:
        indices, feature_thresholds, minimum, policy_threshold = EXPECTED_RULES[
            rule.rule_id
        ]
        config = rule.condition.config
        assert config["feature_indices"] == indices
        assert config["feature_threshold"] == 1.0
        assert config["feature_thresholds"] == feature_thresholds
        assert config["feature_group_rule"] == "cumulative_fraction"
        assert config["min_feature_fraction"] == minimum
        assert config["threshold"] == policy_threshold
        assert rule.candidate_action == "refuse"
        assert profile.action_for_rule(rule.rule_id) == "refuse"


def test_qwen_transcoder_cumulative_fraction_matches_calibrated_outcome():
    profile = load_profile(PROFILE_PATH)
    illegal_indices, illegal_thresholds, _, _ = EXPECTED_RULES["illegal_activity"]
    artifact = FeatureTableTranscoder(
        {index: column for column, index in enumerate(illegal_indices)}
    )
    evaluator, bound_backends = _evaluator(profile, artifact)
    evaluator.add("request")

    activations = []
    for feature_count in (1, 2, 3):
        values = torch.zeros(1, len(illegal_indices))
        for position in range(feature_count):
            values[0, position] = illegal_thresholds[position]
        activations.append(values)

    decisions = [
        evaluator.evaluate(["request"], activation, [step])[0]
        for step, activation in enumerate(activations, start=1)
    ]

    assert [decision.action for decision in decisions] == ["allow", "allow", "refuse"]
    assert decisions[-1].params["rule_id"] == "illegal_activity"
    assert decisions[-1].params["score"] == 0.6
    assert decisions[-1].params["threshold"] == 0.6
    assert all(backend.threshold == 1.0 for backend in bound_backends)


def test_qwen_transcoder_split_copyright_rules_match_merged_bundle_semantics():
    profile = load_profile(PROFILE_PATH)
    artifact = FeatureTableTranscoder({66477: 0, 5332: 1})
    evaluator, _bound_backends = _evaluator(profile, artifact)
    request_ids = ("neither", "book", "lyrics", "both")
    for request_id in request_ids:
        evaluator.add(request_id)

    book_threshold = EXPECTED_RULES["book_copyright"][1][0]
    lyrics_threshold = EXPECTED_RULES["lyrics_copyright"][1][0]
    activations = torch.tensor(
        [
            [0.0, 0.0],
            [book_threshold, 0.0],
            [0.0, lyrics_threshold],
            [book_threshold, lyrics_threshold],
        ]
    )
    decisions = evaluator.evaluate(list(request_ids), activations, [1, 1, 1, 1])

    merged_scores = (0.0, 0.5, 0.5, 1.0)
    assert [decision.action == "refuse" for decision in decisions] == [
        score >= 0.5 for score in merged_scores
    ]
    assert decisions[1].params["rule_id"] == "book_copyright"
    assert decisions[2].params["rule_id"] == "lyrics_copyright"
    assert decisions[3].params["rule_id"] == "book_copyright"
