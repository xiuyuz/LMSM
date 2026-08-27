from __future__ import annotations

from lmsm.state import StepDraft, StepState


class BaseRuntime:
    def start(self, prompt: str) -> StepState:
        raise NotImplementedError

    def inspect_step(self, state: StepState) -> StepDraft:
        raise NotImplementedError

    def commit(self, state: StepState, draft: StepDraft) -> StepState:
        raise NotImplementedError

    def decode(self, state: StepState) -> str:
        raise NotImplementedError
