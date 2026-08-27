"""Public builders for deployment profiles."""

from __future__ import annotations

import json
from pathlib import Path

from lmsm.aggregators.max_score import MaxScoreAggregator
from lmsm.backends.dense_probe import (
    load_anytime_probes,
    load_checkpoint_rules,
)
from lmsm.backends.shared import BoundFeatureBackend, SharedFeatureBackend
from lmsm.control_plane import BackendBinding, DeploymentProfile
from lmsm.engine import GuardEngine
from lmsm.loaders.dense_probe_backend_loader import load_dense_probe_backend
from lmsm.loaders.sae_loader import load_sae
from lmsm.integrations.transformers import TransformersRuntime
from lmsm.policies.threshold import ThresholdPolicy
from lmsm.policy_evaluator import (
    AnytimePolicyEvaluator,
    CheckpointPolicyEvaluator,
    SharedFeaturePolicyEvaluator,
)


def build_backend_artifact(
    binding: BackendBinding,
    project_root: str | Path,
    device: str = "cpu",
):
    """Load the artifact referenced by one backend binding."""

    if binding.backend_type == "dense_probe":
        report_path = _required_report_path(binding, project_root)
        categories = [
            str(value) for value in (binding.artifact or {}).get("categories", [])
        ]
        if categories:
            return load_dense_probe_backend(report_path, categories, device=device)
        return json.loads(report_path.read_text(encoding="utf-8"))

    if binding.backend_type in {"sae", "transcoder"}:
        artifact = dict(binding.artifact or {})
        path = artifact.get("path")
        resolved_path = _resolve_path(project_root, path) if path is not None else None
        return load_sae(
            resolved_path,
            source=str(artifact.get("source", "disk")),
            release=artifact.get("release"),
            sae_id=artifact.get("sae_id"),
            repo_id=artifact.get("repo_id"),
            filename=artifact.get("filename"),
            device=device,
        )

    raise ValueError(f"unsupported backend type: {binding.backend_type}")


def build_shared_feature_backends(
    profile: DeploymentProfile,
    binding: BackendBinding,
    artifact,
    *,
    selected: bool = True,
) -> tuple[SharedFeatureBackend, list[BoundFeatureBackend]]:
    """Bind active rules to one shared feature computation in policy order."""

    active_rules = [
        rule for rule in profile.active_rules if rule.binding_id == binding.binding_id
    ]
    if not active_rules:
        raise ValueError(f"no active rules use backend binding {binding.binding_id!r}")

    rule_configs = []
    for rule in active_rules:
        config = {**binding.config, **rule.condition.config}
        feature_indices = [
            int(index) for index in config.get("feature_indices", [])
        ]
        if not feature_indices:
            raise ValueError(f"rule {rule.rule_id!r} requires feature_indices")
        rule_configs.append((rule, config, feature_indices))

    shared_backend = SharedFeatureBackend(
        artifact,
        binding.activation_key,
        [
            index
            for _rule, _config, feature_indices in rule_configs
            for index in feature_indices
        ],
        selected=selected,
    )
    bound_backends = [
        BoundFeatureBackend(
            shared_backend=shared_backend,
            name=rule.rule_id,
            category=rule.target_id,
            feature_indices=feature_indices,
            threshold=float(
                config["feature_threshold"]
                if "feature_threshold" in config
                else config["threshold"]
            ),
            feature_thresholds=config.get("feature_thresholds"),
            feature_group_rule=str(config.get("feature_group_rule", "any")),
            min_feature_fraction=float(config.get("min_feature_fraction", 1.0)),
        )
        for rule, config, feature_indices in rule_configs
    ]
    return shared_backend, bound_backends


def build_policy_evaluator(
    profile: DeploymentProfile,
    project_root: str | Path,
    actions_enabled: bool = True,
    device: str = "cpu",
):
    """Build the request-keyed evaluator selected by a deployment profile."""

    active_rules = list(profile.active_rules)
    if not active_rules:
        raise ValueError("the deployment profile has no active rules")
    binding_ids = list(dict.fromkeys(rule.binding_id for rule in active_rules))
    if len(binding_ids) != 1:
        raise ValueError("one policy evaluator requires one shared backend binding")
    binding = profile.binding_for(binding_ids[0])
    artifact = build_backend_artifact(binding, project_root, device=device)

    if binding.backend_type == "dense_probe" and isinstance(artifact, dict):
        target_ids = [rule.target_id for rule in active_rules]
        schedule = profile.policy.schedule.kind
        if schedule == "anytime":
            probes = load_anytime_probes(artifact, target_ids)
            return AnytimePolicyEvaluator(
                probes,
                profile,
                actions_enabled=actions_enabled,
            )
        if schedule == "checkpoint":
            rules = load_checkpoint_rules(artifact, target_ids)
            return CheckpointPolicyEvaluator(
                rules,
                profile,
                actions_enabled=actions_enabled,
            )
        raise ValueError(f"unsupported dense-probe schedule: {schedule}")

    if binding.backend_type in {"dense_probe", "sae", "transcoder"}:
        shared_backend, bound_backends = build_shared_feature_backends(
            profile,
            binding,
            artifact,
            selected=True,
        )
        return SharedFeaturePolicyEvaluator(
            shared_backend,
            bound_backends,
            MaxScoreAggregator(),
            ThresholdPolicy(actions_enabled=actions_enabled),
            profile,
        )

    raise ValueError(f"unsupported evaluator backend: {binding.backend_type}")


def build_transformers_engine(
    model,
    tokenizer,
    profile: DeploymentProfile,
    project_root: str | Path,
    *,
    actions_enabled: bool = True,
    device: str | None = None,
    runtime_cls=TransformersRuntime,
    runtime_kwargs: dict | None = None,
    step_logger=None,
) -> GuardEngine:
    """Build the reference Hugging Face engine for a deployment profile."""

    evaluator_device = device or _infer_model_device(model) or "cpu"
    evaluator = build_policy_evaluator(
        profile,
        project_root,
        actions_enabled=actions_enabled,
        device=evaluator_device,
    )
    activation_keys = list(
        dict.fromkeys(
            profile.binding_for(rule.binding_id).activation_key
            for rule in profile.active_rules
        )
    )
    kwargs = dict(runtime_kwargs or {})
    kwargs["target_activations"] = activation_keys
    runtime = runtime_cls(model, tokenizer, **kwargs)
    return GuardEngine(
        runtime,
        evaluator,
        profile,
        step_logger=step_logger,
    )


def _required_report_path(
    binding: BackendBinding,
    project_root: str | Path,
) -> Path:
    if binding.report_path is None:
        raise ValueError(f"backend binding {binding.binding_id!r} has no report path")
    return _resolve_path(project_root, binding.report_path)


def _resolve_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def _infer_model_device(model) -> str | None:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        device = getattr(getattr(embeddings, "weight", None), "device", None)
        if device is not None:
            return str(device)

    device = getattr(model, "device", None)
    if device is not None:
        return str(device)

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter = next(parameters())
        except (StopIteration, TypeError):
            return None
        device = getattr(parameter, "device", None)
        if device is not None:
            return str(device)
    return None


__all__ = [
    "build_backend_artifact",
    "build_transformers_engine",
    "build_policy_evaluator",
    "build_shared_feature_backends",
]
