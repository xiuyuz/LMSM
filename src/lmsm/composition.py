"""Fixed-OR composition of deny-only policies that read one activation.

Composition adds rules.  A composed profile is a YAML file whose ``compose:``
list names member profiles.  Each member keeps its targets, backend binding,
rules, schedule and evaluator.  It contributes its active rules, or a listed
subset, under its member name, for example ``probe/chemical_biological`` and
``transcoder/chemical_biological``.

The composed policy is ``fixed_or``.  A request is refused by the first rule
that fires, and among rules that fire at the same step the one listed first
wins.  Members keep their listed order, and each member keeps its own rule
order.

Every active rule must be deny-only, and all members must read the same
activation, so the runtime keeps one capture.  Execution accounting covers the
union of the members' declared observations and evaluation points.  Execution
states and the release rule are the same as for a single policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from lmsm.control_plane import (
    ActionMapping,
    Composition,
    DeploymentProfile,
    PolicyBundle,
    RuleLibrary,
    Schedule,
)
from lmsm.execution_contract import ExecutionRequirements, ObservationWindow
from lmsm.state import Decision


@dataclass(frozen=True)
class Member:
    """One member policy: its name (the namespace of its rules) and its profile."""

    name: str
    profile: DeploymentProfile


def is_composed_profile(path: str | Path) -> bool:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return isinstance(payload, dict) and "compose" in payload


def load_composed_profile(path: str | Path) -> tuple[DeploymentProfile, tuple[Member, ...]]:
    """Load a ``compose:`` profile: the composed profile and its members.

    Member profile paths are relative to the composed file.  A member listed
    with ``active_rules: []`` contributes no rule and is left out.
    """

    from lmsm.loaders.profile_loader import load_profile

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    members = []
    for item in payload["compose"]:
        profile = load_profile(path.parent / item["profile"])
        if "active_rules" in item:
            selected = tuple(str(rule_id) for rule_id in item["active_rules"])
            for rule_id in selected:
                profile.rule_for(rule_id)
            profile = profile.with_active_rules(selected)
        if profile.policy.active_rule_ids:
            members.append(Member(str(item["name"]), profile))
    if not members:
        raise ValueError("a composed profile needs at least one active rule")
    return compose_profiles(str(payload["name"]), str(payload.get("version", "1")), members), tuple(members)


def compose_profiles(name: str, version: str, members: list[Member]) -> DeploymentProfile:
    """The composed profile: the union of the members' active rules, namespaced."""

    names = [member.name for member in members]
    if len(set(names)) != len(names) or any("/" in name for name in names):
        raise ValueError(f"member names must be unique and contain no '/': {names}")
    targets, bindings, rules, active = {}, {}, [], []
    keys, dispositions = set(), set()
    for member in members:
        profile = member.profile
        dispositions.add(profile.policy.not_reached_disposition)
        for rule in profile.active_rules:
            if profile.action_for_rule(rule.rule_id) != "refuse":
                raise ValueError(f"composition requires deny-only rules; {member.name}/{rule.rule_id} is not")
            target = profile.target_for(rule.target_id)
            if targets.setdefault(target.target_id, target) != target:
                raise ValueError(f"target {target.target_id!r} differs between members")
            binding = profile.binding_for(rule.binding_id)
            if bindings.setdefault(binding.binding_id, binding) != binding:
                raise ValueError(f"backend binding {binding.binding_id!r} differs between members")
            keys.add(binding.activation_key)
            composed_id = f"{member.name}/{rule.rule_id}"
            rules.append(replace(rule, rule_id=composed_id))
            active.append(composed_id)
    if len(keys) != 1:
        raise ValueError(f"composed members must read one activation (one capture); got {sorted(keys)}")
    if len(dispositions) != 1:
        raise ValueError(f"composed members declare different not-reached dispositions: {sorted(dispositions)}")
    return DeploymentProfile(
        name=name,
        targets=tuple(targets.values()),
        backend_bindings=tuple(bindings.values()),
        rule_library=RuleLibrary(
            version="+".join(f"{m.name}:{m.profile.rule_library.version}" for m in members),
            rules=tuple(rules),
        ),
        policy=PolicyBundle(
            name=name,
            version=version,
            active_rule_ids=tuple(active),
            schedule=Schedule(kind="composed"),
            composition=Composition(kind="fixed_or", tie_break="active_rule_order"),
            action_mapping=ActionMapping({"refuse": "refuse"}),
            not_reached_disposition=dispositions.pop(),
        ),
    )


def union_requirements(profile: DeploymentProfile, parts: list[tuple[str, ExecutionRequirements]]) -> ExecutionRequirements:
    """Required observations and evaluation points of the union of the members.

    The union is exact.  When every member has fixed evaluation points, the
    union keeps them as a set.  When some member evaluates at every one of its
    observations, every fixed member's windows and evaluation points must lie
    inside that member's running windows.  Every union observation is then an
    evaluation point.  Any other combination is refused.
    """

    windows: list[ObservationWindow] = []
    for _, requirements in parts:
        for window in requirements.windows:
            if window not in windows:
                windows.append(window)
    running = [requirements for _, requirements in parts if requirements.evaluation_steps is None]
    if not running:
        evaluation_steps: tuple[int, ...] | None = tuple(
            sorted({step for _, requirements in parts for step in requirements.evaluation_steps})
        )
    else:
        running_windows = [window for requirements in running for window in requirements.windows]

        def covered(first: int, last: int | None) -> bool:
            return any(
                window.first_step <= first
                and (window.last_step is None or (last is not None and last <= window.last_step))
                for window in running_windows
            )

        for _, requirements in parts:
            if requirements.evaluation_steps is None:
                continue
            spans = [(w.first_step, w.last_step) for w in requirements.windows]
            spans += [(step, step) for step in requirements.evaluation_steps]
            if not all(covered(first, last) for first, last in spans):
                raise ValueError(
                    "the union of the members' evaluation points is not expressible: a fixed "
                    "schedule observes or evaluates outside every running schedule's windows"
                )
        evaluation_steps = None
    return ExecutionRequirements(
        policy_name=str(profile.policy.name),
        policy_version=str(profile.policy.version),
        rule_library_version=str(profile.rule_library.version),
        schedule={"kind": "composed", "members": {name: dict(r.schedule) for name, r in parts}},
        windows=tuple(windows),
        evaluation_steps=evaluation_steps,
        not_reached_disposition=profile.policy.not_reached_disposition,
        coverage="; ".join(f"{name}: {r.coverage}" for name, r in parts),
    )


class ComposedPolicyEvaluator:
    """Evaluate member policies on one shared activation and OR their decisions.

    At each step, every member receives exactly the rows that its own schedule
    observes, through its own ``evaluate``.  Member evaluators always decide.
    This evaluator applies ``actions_enabled``, so observe-only runs still
    record each member's first trigger (step and rule) for analysis.
    """

    def __init__(self, members: list[tuple[str, Any]], profile: DeploymentProfile, *, actions_enabled: bool = True):
        self.members = list(members)
        self.profile = profile
        self.actions_enabled = bool(actions_enabled)
        self.order = {rule_id: index for index, rule_id in enumerate(profile.policy.active_rule_ids)}
        self.observes_prompt = any(m.observes_prompt for _, m in self.members)
        self.observes_decode = any(m.observes_decode for _, m in self.members)
        limits = [m.max_decode_step for _, m in self.members if m.observes_decode]
        self.max_decode_step = None if any(limit is None for limit in limits) else max(limits, default=None)
        self.first_triggers: dict[str, dict[str, dict[str, Any]]] = {}

    # -- request lifecycle -------------------------------------------------
    def add(self, request_id: str) -> None:
        for _, member in self.members:
            member.add(request_id)
        self.first_triggers[request_id] = {}

    def remove(self, request_id: str) -> None:
        for _, member in self.members:
            member.remove(request_id)
        self.first_triggers.pop(request_id, None)

    def needs_prompt(self, request_id: str) -> bool:
        return any(m.observes_prompt and m.needs_prompt(request_id) for _, m in self.members)

    def execution_requirements(self) -> ExecutionRequirements:
        return union_requirements(
            self.profile, [(name, m.execution_requirements()) for name, m in self.members]
        )

    # -- evaluation --------------------------------------------------------
    @staticmethod
    def _observes(member: Any, request_id: str, step: int) -> bool:
        if step == 0:
            return bool(member.observes_prompt and member.needs_prompt(request_id))
        limit = member.max_decode_step
        return bool(member.observes_decode and (limit is None or step <= limit))

    def evaluate(self, request_ids: list[str], activations, steps: list[int]) -> list[Decision]:
        candidates: list[list[tuple[int, str, Decision]]] = [[] for _ in request_ids]
        for name, member in self.members:
            rows = [i for i, (rid, step) in enumerate(zip(request_ids, steps)) if self._observes(member, rid, step)]
            if not rows:
                continue
            decisions = member.evaluate(
                [request_ids[i] for i in rows], activations[rows], [steps[i] for i in rows]
            )
            for i, decision in zip(rows, decisions, strict=True):
                if decision.action == "allow":
                    continue
                composed_id = f"{name}/{decision.params['rule_id']}"
                candidates[i].append((self.order[composed_id], name, decision))
                self.first_triggers[request_ids[i]].setdefault(
                    name,
                    {"step": int(steps[i]), "rule_id": composed_id, "score": decision.params.get("score")},
                )
        out = []
        for i, found in enumerate(candidates):
            if not found or not self.actions_enabled:
                out.append(Decision(action="allow", reason="no composed rule fired" if not found else "actions disabled"))
                continue
            found.sort(key=lambda item: item[0])
            _, name, decision = found[0]
            params = dict(decision.params)
            params["rule_id"] = f"{name}/{decision.params['rule_id']}"
            params["backend"] = name
            params["co_fired"] = [f"{n}/{d.params['rule_id']}" for _, n, d in found[1:]]
            out.append(Decision(action=decision.action, params=params, reason=f"{params['rule_id']} fired first"))
        return out

    # -- records -----------------------------------------------------------
    def scores_for(self, request_id: str) -> dict[str, Any]:
        return {name: m.scores_for(request_id) for name, m in self.members}

    def score_steps_for(self, request_id: str) -> dict[str, int]:
        """First firing step of each member (its first-firing rule), on this request's own text."""

        return {t["rule_id"]: t["step"] for t in self.first_triggers.get(request_id, {}).values()}

    def actions_for(self, request_id: str) -> dict[str, bool]:
        return {t["rule_id"]: True for t in self.first_triggers.get(request_id, {}).values()}

    def nonfinite_step_for(self, request_id: str) -> int | None:
        """First step at which any member computed a NaN or infinite value."""

        steps = []
        for _, member in self.members:
            step_for = getattr(member, "nonfinite_step_for", None)
            step = step_for(request_id) if step_for is not None else None
            if step is not None:
                steps.append(step)
        return min(steps) if steps else None

    def feature_maxima_for(self, request_id: str) -> dict[str, dict[int, float]]:
        merged: dict[str, dict[int, float]] = {}
        for name, member in self.members:
            for category, maxima in member.feature_maxima_for(request_id).items():
                merged[f"{name}/{category}"] = dict(maxima)
        return merged


def build_composed_evaluator(path: str | Path, project_root: str | Path, *, actions_enabled: bool = True, device: str = "cpu"):
    """The composed profile and its evaluator (each member built by ``build_policy_evaluator``)."""

    from lmsm.build import build_policy_evaluator

    profile, members = load_composed_profile(path)
    evaluators = [
        (member.name, build_policy_evaluator(member.profile, project_root, actions_enabled=True, device=device))
        for member in members
    ]
    return profile, ComposedPolicyEvaluator(evaluators, profile, actions_enabled=actions_enabled)


__all__ = [
    "ComposedPolicyEvaluator",
    "Member",
    "build_composed_evaluator",
    "compose_profiles",
    "is_composed_profile",
    "load_composed_profile",
    "union_requirements",
]
