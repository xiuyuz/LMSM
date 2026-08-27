from __future__ import annotations

from lmsm.state import BackendSignal, StepDraft


class BaseBackend:
    """Observe a draft token and emit a normalized risk signal in [0, 1]."""

    name: str
    category: str

    def observe(self, draft: StepDraft) -> BackendSignal:
        raise NotImplementedError

    def reset_state(self) -> None:
        """Reset any per-row cumulative state. Called once before each new generation."""
