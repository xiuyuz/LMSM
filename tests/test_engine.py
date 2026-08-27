from pathlib import Path

import torch

from lmsm.engine import GuardEngine
from lmsm.loaders.profile_loader import load_profile
from lmsm.state import Decision, StepDraft, StepState


ROOT = Path(__file__).resolve().parents[1]


class CandidateRuntime:
    tokenizer = None
    model = None

    def __init__(self, trace, activation_key):
        self.trace = trace
        self.activation_key = activation_key

    def start(self, prompt):
        self.trace.append("start")
        return StepState(0, [[1]], [[]], metadata={"prompt": prompt})

    def inspect_step(self, state):
        self.trace.append(f"inspect:{state.step_idx}")
        return StepDraft(
            step_idx=state.step_idx,
            input_ids=state.input_ids,
            generated_ids=state.generated_ids,
            logits=torch.tensor([[0.0, 1.0]]),
            next_token_id=[[9]],
            activations={self.activation_key: torch.tensor([[1.0]])},
        )

    def commit(self, state, draft):
        self.trace.append(f"commit:{state.step_idx}")
        state.generated_ids = [state.generated_ids[0] + draft.next_token_id[0]]
        state.step_idx += 1
        return state

    def decode(self, state):
        self.trace.append("decode")
        return state.metadata.get("output_text_override", "9")


class SecondCandidateRefusal:
    observes_prompt = True
    observes_decode = True
    max_decode_step = None

    def __init__(self, trace):
        self.trace = trace

    def add(self, _request_id):
        self.trace.append("add")

    def remove(self, _request_id):
        self.trace.append("remove")

    def evaluate(self, _request_ids, _activations, steps):
        step = steps[0]
        self.trace.append(f"evaluate:{step}")
        if step == 0:
            return [Decision("allow")]
        return [
            Decision(
                "refuse",
                {"rule_id": "chemical_biological"},
                "candidate refused",
            )
        ]

    def scores_for(self, _request_id):
        return {}

    def actions_for(self, _request_id):
        return {}


def test_hf_engine_inspects_candidate_before_commit_and_refuses_uncommitted_token():
    profile = load_profile(ROOT / "profiles/lmsm_anytime.yaml")
    activation_key = profile.backend_bindings[0].activation_key
    trace = []
    engine = GuardEngine(
        CandidateRuntime(trace, activation_key),
        SecondCandidateRefusal(trace),
        profile,
    )

    output = engine.generate("prompt", max_new_tokens=3)

    assert output == "I can't help with that."
    assert trace == [
        "add",
        "start",
        "inspect:0",
        "evaluate:0",
        "commit:0",
        "inspect:1",
        "evaluate:1",
        "decode",
        "remove",
    ]
