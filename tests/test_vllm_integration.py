from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from lmsm.loaders.profile_loader import load_profile
from lmsm.state import Decision
from lmsm.integrations import vllm as runtime


ROOT = Path(__file__).resolve().parents[1]


class DeterministicEvaluator:
    observes_prompt = False
    observes_decode = True
    max_decode_step = None

    def __init__(self):
        base_profile = load_profile(ROOT / "profiles/lmsm_anytime.yaml")
        binding = replace(
            base_profile.backend_bindings[0],
            activation_key="pre:decoder.blocks.3.feed_forward",
            config={"hidden_width": 4},
        )
        self.profile = replace(
            base_profile,
            backend_bindings=(binding,),
        )
        self.active = set()
        self.scores = {}
        self.actions = {}

    def add(self, request_id):
        self.active.add(request_id)
        self.scores[request_id] = {}
        self.actions[request_id] = {}

    def remove(self, request_id):
        self.active.discard(request_id)

    def needs_prompt(self, _request_id):
        return False

    def evaluate(self, request_ids, activations, _steps):
        decisions = []
        for row, request_id in enumerate(request_ids):
            score = float(activations[row, 0])
            crossed = score > 0
            self.scores[request_id] = {"chemical_biological": score}
            self.actions[request_id] = {"chemical_biological": crossed}
            decisions.append(
                Decision(
                    "refuse",
                    {
                        "rule_id": "chemical_biological",
                        "target_id": "chemical_biological",
                        "score": score,
                        "threshold": 0.5,
                    },
                    "chemical_biological crossed its LMSM-Anytime threshold",
                )
                if crossed
                else Decision("allow")
            )
        return decisions

    def feature_maxima_for(self, _request_id):
        return {}

    def scores_for(self, request_id):
        return dict(self.scores[request_id])

    def score_steps_for(self, _request_id):
        return {}

    def actions_for(self, request_id):
        return dict(self.actions[request_id])


def _params(request_id):
    return SimpleNamespace(
        extra_args={runtime.REQUEST_ID_EXTRA_ARG: request_id},
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )


def _update(*, batch_size, removed=(), added=(), moved=()):
    return SimpleNamespace(
        batch_size=batch_size,
        removed=list(removed),
        added=list(added),
        moved=list(moved),
    )


def test_batch_policy_state_tracks_moves_and_selectively_buffers_refusal():
    evaluator = DeterministicEvaluator()
    state = runtime.STATE
    state.configure(evaluator, [91, 92])
    state.eos_token_id = 3
    output_a = [7]
    output_b = [8]

    state.update_batch(
        _update(
            batch_size=2,
            added=[
                (0, _params("request-a"), [10], output_a),
                (1, _params("request-b"), [11], output_b),
            ],
        )
    )
    state.update_batch(
        _update(
            batch_size=2,
            moved=[(0, 1, SimpleNamespace(name="SWAP"))],
        )
    )
    assert state.rows[0].request_id == "request-b"
    assert state.rows[1].request_id == "request-a"
    assert state.activation_key == "pre:decoder.blocks.3.feed_forward"
    assert state.expected_hidden_width == 4

    state.begin_forward([1, 1])
    hidden = torch.zeros(2, state.expected_hidden_width)
    hidden[1, 0] = 1.0
    state.capture(hidden)
    logits = torch.ones(2, 4)
    state.apply_actions(logits)

    assert torch.equal(logits[0], torch.ones(4))
    assert logits[1].argmax().item() == 3
    assert torch.isneginf(logits[1, :3]).all()

    record = state.record_for("request-a")
    assert record is not None
    assert record.to_dict() == {
        "request_id": "request-a",
        "step": 1,
        "action": "refuse",
        "rule_id": "chemical_biological",
        "target_id": "chemical_biological",
        "score": 1.0,
        "threshold": 0.5,
        "policy_name": "LMSM-Anytime",
        "policy_version": "1",
        "rule_library_version": "1",
        "active_rule_ids": list(evaluator.profile.policy.active_rule_ids),
        "ordinary_model_prefix_tokens": [7],
        "forced_termination_token": 3,
        "fixed_refusal_tokens": [91, 92],
        "admitted_prefix_tokens": [7],
        "released_output_tokens": [91, 92],
    }
    assert runtime.buffered_output("request-a", [7, 99])["released_output_tokens"] == [
        91,
        92,
    ]

    fresh_output = []
    state.update_batch(
        _update(
            batch_size=1,
            removed=[0, 1],
            added=[(0, _params("request-c"), [12], fresh_output)],
        )
    )
    assert state.rows[0].request_id == "request-c"
    assert state.rows[0].actioned is False
    assert evaluator.active == {"request-c"}
    assert state.record_for("request-c") is None
    state.reset()


def test_matched_empty_extension_leaves_logits_unchanged():
    processor = runtime.MatchedEmptyExtensionProcessor(None, None, False)
    logits = torch.tensor([[1.0, 2.0]])

    assert processor.apply(logits) is logits
