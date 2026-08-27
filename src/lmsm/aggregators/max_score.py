from __future__ import annotations

from collections import defaultdict

from lmsm.aggregator import BaseRiskAggregator
from lmsm.control_plane import DeploymentProfile
from lmsm.state import BackendSignal, RiskAssessment


class MaxScoreAggregator(BaseRiskAggregator):
    def aggregate(
        self,
        signals: list[BackendSignal],
        profile: DeploymentProfile,
    ) -> list[RiskAssessment]:
        grouped: dict[str, list[BackendSignal]] = defaultdict(list)
        active_targets = {rule.target_id for rule in profile.active_rules}
        for signal in signals:
            if signal.category in active_targets:
                grouped[signal.category].append(signal)

        assessments = []
        for rule in profile.active_rules:
            target_signals = grouped.get(rule.target_id, [])
            score = _mean_signal_score(target_signals)
            threshold = float(rule.condition.config.get("threshold", 1.0))
            level = "medium" if score >= threshold else "none"
            assessments.append(
                RiskAssessment(
                    category=rule.target_id,
                    score=score,
                    level=level,
                    triggered=level == "medium",
                    supporting_backends=[
                        signal.backend
                        for signal in target_signals
                        if signal.triggered
                    ],
                    info={
                        "threshold": threshold,
                        "num_backend_signals": len(target_signals),
                        "num_triggered": sum(
                            1 for signal in target_signals if signal.triggered
                        ),
                    },
                )
            )
        return assessments


def _mean_signal_score(signals: list[BackendSignal]) -> float:
    if not signals:
        return 0.0
    return sum(float(signal.score) for signal in signals) / len(signals)
