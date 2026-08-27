from __future__ import annotations

from lmsm.control_plane import DeploymentProfile
from lmsm.policy import BaseDecisionPolicy
from lmsm.state import Decision, RiskAssessment


class ThresholdPolicy(BaseDecisionPolicy):
    """Compose triggered rules in the profile's declared active-rule order."""

    def __init__(self, *, actions_enabled: bool = True) -> None:
        self.actions_enabled = actions_enabled

    def decide(
        self,
        risks: list[RiskAssessment],
        _state,
        profile: DeploymentProfile,
    ) -> Decision:
        risks_by_target = {risk.category: risk for risk in risks}
        candidates = [
            rule.rule_id
            for rule in profile.active_rules
            if (risk := risks_by_target.get(rule.target_id)) is not None
            and risk.triggered
        ]
        selected_rule_id = profile.policy.compose(candidates)
        if selected_rule_id is None:
            return Decision(action="allow", reason="no active rule triggered")
        rule = profile.rule_for(selected_rule_id)
        risk = risks_by_target[rule.target_id]
        if not self.actions_enabled:
            return Decision(action="allow", reason="actions disabled")
        return Decision(
            action=profile.action_for_rule(rule.rule_id),
            params={
                "rule_id": rule.rule_id,
                "target_id": rule.target_id,
                "category": rule.target_id,
                "risk_level": risk.level,
                "score": risk.score,
                "threshold": risk.info["threshold"],
            },
            reason=f"{rule.rule_id} crossed its {profile.policy.name} threshold",
        )
