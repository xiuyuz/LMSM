from __future__ import annotations

from dataclasses import asdict

from lmsm.control_plane import DeploymentProfile
from lmsm.enforcement import BufferedOutputGate
from lmsm.state import Decision
from lmsm.utils.values import last_token_representation, to_row_vector


class GuardEngine:
    """Single-request HF generation mediated by a deployment profile."""

    def __init__(
        self,
        runtime,
        policy_evaluator,
        profile: DeploymentProfile,
        *,
        enforcement_gate=None,
        step_logger=None,
    ) -> None:
        self.runtime = runtime
        self.policy_evaluator = policy_evaluator
        self.profile = profile
        self.enforcement_gate = enforcement_gate or BufferedOutputGate()
        self.step_logger = step_logger
        self.activation_key = _active_activation_key(profile)
        self._request_number = 0

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        request_id = f"hf-{self._request_number}"
        self._request_number += 1
        self.policy_evaluator.add(request_id)
        try:
            state = self.runtime.start(prompt)
            state.metadata.setdefault("terminated", False)
            stop_token_ids = _collect_stop_token_ids(self.runtime)

            for _ in range(max_new_tokens):
                if state.metadata.get("terminated", False):
                    break

                draft = self.runtime.inspect_step(state)
                decision = self._evaluate(request_id, draft)
                state = self.enforcement_gate.apply(state, decision)
                state.cache["last_decision"] = decision
                state.cache["last_draft"] = draft
                self._log_step(request_id, draft, state, decision)

                if state.metadata.get("terminated", False):
                    break

                state = self.runtime.commit(state, draft)
                if _is_stop_token(draft.next_token_id, stop_token_ids):
                    state.metadata["terminated"] = True
                    state.metadata["termination_reason"] = "eos"
                    break

            return self.runtime.decode(state)
        finally:
            self.policy_evaluator.remove(request_id)

    def _evaluate(self, request_id: str, draft) -> Decision:
        step = int(draft.step_idx)
        if step == 0:
            should_observe = bool(self.policy_evaluator.observes_prompt)
        else:
            should_observe = bool(self.policy_evaluator.observes_decode)

        max_decode_step = self.policy_evaluator.max_decode_step
        if not should_observe or (
            step > 0 and max_decode_step is not None and step > max_decode_step
        ):
            return Decision(action="allow")

        activation = (draft.activations or {}).get(self.activation_key)
        if activation is None:
            raise KeyError(f"Missing activation for {self.activation_key}")
        endpoint = to_row_vector(last_token_representation(activation))
        return self.policy_evaluator.evaluate([request_id], endpoint, [step])[0]

    def _log_step(self, request_id: str, draft, state, decision: Decision) -> None:
        if self.step_logger is None:
            return
        self.step_logger.log_step(
            {
                "step": draft.step_idx,
                "policy": self.profile.policy.name,
                "decision": asdict(decision),
                "scores": self.policy_evaluator.scores_for(request_id),
                "actions": self.policy_evaluator.actions_for(request_id),
                "terminated": bool(state.metadata.get("terminated", False)),
            }
        )

def _active_activation_key(profile: DeploymentProfile) -> str:
    keys = []
    for rule in profile.active_rules:
        key = profile.binding_for(rule.binding_id).activation_key
        if key not in keys:
            keys.append(key)
    if len(keys) != 1:
        raise ValueError("the HF engine requires one shared activation site")
    return keys[0]


def _collect_stop_token_ids(runtime) -> set[int]:
    ids: set[int] = set()

    def _add(value):
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _add(item)
            return
        try:
            ids.add(int(value))
        except (TypeError, ValueError):
            pass

    tokenizer = getattr(runtime, "tokenizer", None)
    if tokenizer is not None:
        _add(getattr(tokenizer, "eos_token_id", None))

    model = getattr(runtime, "model", None)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        _add(getattr(generation_config, "eos_token_id", None))
    return ids


def _is_stop_token(next_token_id, stop_token_ids: set[int]) -> bool:
    if not stop_token_ids or next_token_id is None:
        return False
    if hasattr(next_token_id, "tolist"):
        return _is_stop_token(next_token_id.tolist(), stop_token_ids)
    if isinstance(next_token_id, (list, tuple)):
        return any(_is_stop_token(item, stop_token_ids) for item in next_token_id)
    try:
        return int(next_token_id) in stop_token_ids
    except (TypeError, ValueError):
        return False
