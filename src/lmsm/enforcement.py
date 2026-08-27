"""State-changing actions applied after LMSM policy evaluation."""

from __future__ import annotations

from lmsm.state import Decision


class BufferedOutputGate:
    """Apply one decision to a buffered single-request generation state."""

    def __init__(self, refusal_text: str = "I can't help with that.") -> None:
        self.refusal_text = refusal_text

    def apply(self, state, decision: Decision):
        if decision.action == "allow":
            return state
        if decision.action == "terminate":
            state.metadata["output_text_override"] = ""
        elif decision.action == "refuse":
            state.metadata["output_text_override"] = decision.params.get(
                "refusal_text", self.refusal_text
            )
        else:
            raise KeyError(f"Unknown action: {decision.action}")

        state.metadata["terminated"] = True
        state.metadata["termination_reason"] = decision.reason
        return state


__all__ = ["BufferedOutputGate"]
