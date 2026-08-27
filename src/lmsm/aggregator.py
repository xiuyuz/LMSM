from __future__ import annotations

from lmsm.control_plane import DeploymentProfile
from lmsm.state import BackendSignal, RiskAssessment


class BaseRiskAggregator:
    def aggregate(
        self,
        signals: list[BackendSignal],
        profile: DeploymentProfile,
    ) -> list[RiskAssessment]:
        raise NotImplementedError
