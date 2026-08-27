from __future__ import annotations

from lmsm.control_plane import DeploymentProfile
from lmsm.state import Decision, RiskAssessment, StepDraft, StepState


class BaseDecisionPolicy:
    def decide(
        self,
        risks: list[RiskAssessment],
        state: StepState | StepDraft,
        profile: DeploymentProfile,
    ) -> Decision:
        raise NotImplementedError
