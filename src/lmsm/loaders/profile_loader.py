"""Load LMSM deployment profiles from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lmsm.control_plane import (
    ActionMapping,
    BackendBinding,
    Composition,
    DeploymentProfile,
    PolicyBundle,
    RuleCondition,
    RuleLibrary,
    SafetyRule,
    Schedule,
    Target,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value or {})


def load_profile(path: str | Path) -> DeploymentProfile:
    """Load a complete deployment profile from ``path``."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    targets = tuple(
        Target(
            target_id=str(item["id"]),
            behavior=str(item["behavior"]),
            scope=str(item["scope"]),
        )
        for item in payload["targets"]
    )

    backend_bindings = tuple(
        BackendBinding(
            binding_id=str(item["id"]),
            backend_type=str(item["type"]),
            provisioning=str(item["provisioning"]),
            version=str(item["version"]),
            activation_key=str(item["activation_key"]),
            channels=tuple(str(channel) for channel in item["channels"]),
            report_path=(
                str(item["report_path"]) if item.get("report_path") is not None else None
            ),
            artifact=_mapping(item["artifact"]) if item.get("artifact") is not None else None,
            config=_mapping(item.get("config")),
        )
        for item in payload["backend_bindings"]
    )

    library_payload = payload["rule_library"]
    rules = []
    for item in library_payload["rules"]:
        condition_payload = item["condition"]
        rules.append(
            SafetyRule(
                rule_id=str(item["id"]),
                version=str(item["version"]),
                target_id=str(item["target"]),
                binding_id=str(item["binding"]),
                channels=tuple(str(channel) for channel in item["channels"]),
                condition=RuleCondition(
                    kind=str(condition_payload["type"]),
                    channel=(
                        str(condition_payload["channel"])
                        if condition_payload.get("channel") is not None
                        else None
                    ),
                    calibration_key=(
                        str(condition_payload["calibration_key"])
                        if condition_payload.get("calibration_key") is not None
                        else None
                    ),
                    config=_mapping(condition_payload.get("config")),
                ),
                candidate_action=str(item["candidate_action"]),
            )
        )
    rule_library = RuleLibrary(
        version=str(library_payload["version"]),
        rules=tuple(rules),
    )

    policy_payload = payload["policy"]
    schedule_payload = policy_payload["schedule"]
    composition_payload = policy_payload.get("composition", {})
    policy = PolicyBundle(
        name=str(policy_payload["name"]),
        version=str(policy_payload["version"]),
        active_rule_ids=tuple(str(rule_id) for rule_id in policy_payload["active_rules"]),
        schedule=Schedule(
            kind=str(schedule_payload["type"]),
            pooling=(
                str(schedule_payload["pooling"])
                if schedule_payload.get("pooling") is not None
                else None
            ),
            evaluation_step=schedule_payload.get("evaluation_step"),
            window_split_step=schedule_payload.get("window_split_step"),
            max_decode_step=schedule_payload.get("max_decode_step"),
            first_crossing=bool(schedule_payload.get("first_crossing", False)),
        ),
        composition=Composition(
            kind=str(composition_payload.get("type", "fixed_or")),
            tie_break=str(composition_payload.get("tie_break", "active_rule_order")),
        ),
        action_mapping=ActionMapping(
            by_candidate={
                str(candidate): str(action)
                for candidate, action in policy_payload["action_mapping"].items()
            }
        ),
    )

    return DeploymentProfile(
        name=str(payload["name"]),
        targets=targets,
        backend_bindings=backend_bindings,
        rule_library=rule_library,
        policy=policy,
    )
