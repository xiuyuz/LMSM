"""Deployment-profile control-plane types for LMSM."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class Target:
    target_id: str
    behavior: str
    scope: str


@dataclass
class BackendBinding:
    binding_id: str
    backend_type: str
    provisioning: str
    version: str
    activation_key: str
    channels: tuple[str, ...]
    report_path: str | None = None
    artifact: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleCondition:
    kind: str
    channel: str | None = None
    calibration_key: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyRule:
    rule_id: str
    version: str
    target_id: str
    binding_id: str
    channels: tuple[str, ...]
    condition: RuleCondition
    candidate_action: str


@dataclass
class RuleLibrary:
    version: str
    rules: tuple[SafetyRule, ...]

    def rule_for(self, rule_id: str) -> SafetyRule:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        raise KeyError(f"unknown rule: {rule_id}")


@dataclass
class Schedule:
    kind: str
    pooling: str | None = None
    evaluation_step: int | None = None
    window_split_step: int | None = None
    max_decode_step: int | None = None
    first_crossing: bool = False


@dataclass
class Composition:
    kind: str = "fixed_or"
    tie_break: str = "active_rule_order"


@dataclass
class ActionMapping:
    by_candidate: dict[str, str]

    def resolve(self, candidate_action: str) -> str:
        return self.by_candidate[candidate_action]


@dataclass
class PolicyBundle:
    name: str
    version: str
    active_rule_ids: tuple[str, ...]
    schedule: Schedule
    composition: Composition
    action_mapping: ActionMapping

    def compose(self, candidate_rule_ids: list[str]) -> str | None:
        """Select a candidate using this bundle's declared rule order."""

        candidates = set(candidate_rule_ids)
        return next(
            (rule_id for rule_id in self.active_rule_ids if rule_id in candidates),
            None,
        )


@dataclass
class DeploymentProfile:
    name: str
    targets: tuple[Target, ...]
    backend_bindings: tuple[BackendBinding, ...]
    rule_library: RuleLibrary
    policy: PolicyBundle

    @property
    def active_rules(self) -> tuple[SafetyRule, ...]:
        return tuple(self.rule_for(rule_id) for rule_id in self.policy.active_rule_ids)

    @property
    def enabled_targets(self) -> tuple[str, ...]:
        return tuple(rule.target_id for rule in self.active_rules)

    def target_for(self, target_id: str) -> Target:
        for target in self.targets:
            if target.target_id == target_id:
                return target
        raise KeyError(f"unknown target: {target_id}")

    def binding_for(self, binding_id: str) -> BackendBinding:
        for binding in self.backend_bindings:
            if binding.binding_id == binding_id:
                return binding
        raise KeyError(f"unknown backend binding: {binding_id}")

    def rule_for(self, rule_id: str) -> SafetyRule:
        return self.rule_library.rule_for(rule_id)

    def rule_for_target(self, target_id: str) -> SafetyRule:
        for rule in self.rule_library.rules:
            if rule.target_id == target_id:
                return rule
        raise KeyError(f"no rule for target: {target_id}")

    def action_for_rule(self, rule_id: str) -> str:
        rule = self.rule_for(rule_id)
        return self.policy.action_mapping.resolve(rule.candidate_action)

    def with_active_rules(self, rule_ids: list[str] | tuple[str, ...]):
        """Return the deployment with a different active-rule subset."""

        return replace(
            self,
            policy=replace(self.policy, active_rule_ids=tuple(rule_ids)),
        )
