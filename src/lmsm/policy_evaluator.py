"""Request-keyed policy evaluators for built-in LMSM schedules.

Every evaluator also records, per request, the first step at which a value it
computed for the decision (a pooled mean, a probe or fusion score, a feature
activation) was NaN or infinite (``nonfinite_step_for``).  This record does
not change any decision or state.  The serving runtime fails such a request
closed instead of acting on non-finite evidence.
"""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from lmsm.backends.dense_probe import CheckpointRule, DenseProbe
from lmsm.control_plane import DeploymentProfile
from lmsm.execution_contract import ExecutionRequirements, ObservationWindow
from lmsm.state import Decision


class AnytimePolicyEvaluator:
    """Evaluate running-prefix probes and act on their first crossing."""

    observes_prompt = False
    observes_decode = True
    max_decode_step = None

    def __init__(
        self,
        probes: list[DenseProbe],
        profile: DeploymentProfile,
        *,
        actions_enabled: bool = True,
    ) -> None:
        self.probes = list(probes)
        self.profile = profile
        self.actions_enabled = actions_enabled
        self.max_scores_by_id: dict[str, dict[str, float]] = {}
        self.max_steps_by_id: dict[str, dict[str, int]] = {}
        self.actions_by_id: dict[str, dict[str, bool]] = {}
        self.hidden_sums_by_id: dict[str, np.ndarray] = {}
        self.hidden_counts_by_id: dict[str, int] = {}
        self.nonfinite_steps: dict[str, int] = {}

    def add(self, request_id: str) -> None:
        self.max_scores_by_id[request_id] = {}
        self.max_steps_by_id[request_id] = {}
        self.actions_by_id[request_id] = {}
        self.hidden_counts_by_id[request_id] = 0
        self.nonfinite_steps.pop(request_id, None)

    def remove(self, request_id: str) -> None:
        self.max_scores_by_id.pop(request_id, None)
        self.max_steps_by_id.pop(request_id, None)
        self.actions_by_id.pop(request_id, None)
        self.hidden_sums_by_id.pop(request_id, None)
        self.hidden_counts_by_id.pop(request_id, None)
        self.nonfinite_steps.pop(request_id, None)

    def nonfinite_step_for(self, request_id: str) -> int | None:
        """First step whose pooled mean or a probe score was NaN or infinite."""

        return self.nonfinite_steps.get(request_id)

    def needs_prompt(self, request_id: str) -> bool:
        del request_id
        return False

    def execution_requirements(self) -> ExecutionRequirements:
        """Every processed generated token, evaluated after every decode step."""

        return ExecutionRequirements.for_profile(
            self.profile,
            windows=(ObservationWindow("decode", 1, None),),
            evaluation_steps=None,
            coverage=(
                "every processed generated token until the first threshold "
                "crossing; the final generated token is never processed, so it "
                "is released unevaluated; a request that finishes on its first "
                "generated token reaches no evaluation point"
            ),
        )

    def scores_for(self, request_id: str) -> dict[str, float]:
        return dict(self.max_scores_by_id.get(request_id, {}))

    def score_steps_for(self, request_id: str) -> dict[str, int]:
        return dict(self.max_steps_by_id.get(request_id, {}))

    def actions_for(self, request_id: str) -> dict[str, bool]:
        return dict(self.actions_by_id.get(request_id, {}))

    def feature_maxima_for(self, request_id: str) -> dict[str, dict[int, float]]:
        del request_id
        return {}

    def evaluate(
        self,
        request_ids: list[str],
        activations: torch.Tensor,
        steps: list[int],
    ) -> list[Decision]:
        endpoints = activations.detach().to(device="cpu", dtype=torch.float32).numpy()
        pooled = endpoints.copy()
        for row, request_id in enumerate(request_ids):
            current = pooled[row]
            if request_id not in self.hidden_sums_by_id:
                self.hidden_sums_by_id[request_id] = current.copy()
            else:
                self.hidden_sums_by_id[request_id] += current
            self.hidden_counts_by_id[request_id] += 1
            pooled[row] = (
                self.hidden_sums_by_id[request_id]
                / self.hidden_counts_by_id[request_id]
            )

        hidden = pooled.astype(np.float64)
        scores = {probe.target_id: probe.score(hidden) for probe in self.probes}
        finite = np.isfinite(pooled).all(axis=1)
        for values in scores.values():
            finite &= np.isfinite(values)
        decisions = []
        for row, (request_id, step) in enumerate(zip(request_ids, steps, strict=True)):
            if not finite[row]:
                self.nonfinite_steps.setdefault(request_id, step)
            maxima = self.max_scores_by_id[request_id]
            max_steps = self.max_steps_by_id[request_id]
            action_state = self.actions_by_id[request_id]
            triggered: list[tuple[DenseProbe, float]] = []
            for probe in self.probes:
                score = float(scores[probe.target_id][row])
                previous = maxima.get(probe.target_id)
                if previous is None or score > previous:
                    maxima[probe.target_id] = score
                    max_steps[probe.target_id] = step
                crossed_before = action_state.get(probe.target_id, False)
                crossed = bool(maxima[probe.target_id] >= probe.threshold)
                action_state[probe.target_id] = crossed
                if crossed and not crossed_before:
                    triggered.append((probe, maxima[probe.target_id]))
            if not triggered or not self.actions_enabled:
                decisions.append(_allow())
                continue
            probe, score = _selected_trigger(self.profile, triggered)
            decisions.append(
                _rule_decision(
                    self.profile,
                    probe.target_id,
                    score,
                    float(probe.threshold),
                    "running_prefix_first_crossing",
                )
            )
        return decisions


class CheckpointPolicyEvaluator:
    """Evaluate prompt and fixed generation windows once at token 64."""

    observes_prompt = True
    observes_decode = True

    def __init__(
        self,
        rules: list[CheckpointRule],
        profile: DeploymentProfile,
        *,
        actions_enabled: bool = True,
    ) -> None:
        self.rules = list(rules)
        self.profile = profile
        self.actions_enabled = actions_enabled
        schedule = profile.policy.schedule
        self.window_split_step = int(schedule.window_split_step)
        self.evaluation_step = int(schedule.evaluation_step)
        self.max_decode_step = int(schedule.max_decode_step)
        self.prompt_evaluated: set[str] = set()
        self.prompt_scores_by_id: dict[str, dict[str, float]] = {}
        self.window_1_scores_by_id: dict[str, dict[str, float]] = {}
        self.window_2_scores_by_id: dict[str, dict[str, float]] = {}
        self.fusion_scores_by_id: dict[str, dict[str, float]] = {}
        self.action_steps_by_id: dict[str, dict[str, int]] = {}
        self.actions_by_id: dict[str, dict[str, bool]] = {}
        self.window_1_sums_by_id: dict[str, np.ndarray] = {}
        self.window_2_sums_by_id: dict[str, np.ndarray] = {}
        self.window_1_counts_by_id: dict[str, int] = {}
        self.window_2_counts_by_id: dict[str, int] = {}
        self.nonfinite_steps: dict[str, int] = {}

    def add(self, request_id: str) -> None:
        self.prompt_scores_by_id[request_id] = {}
        self.window_1_scores_by_id[request_id] = {}
        self.window_2_scores_by_id[request_id] = {}
        self.fusion_scores_by_id[request_id] = {}
        self.action_steps_by_id[request_id] = {}
        self.actions_by_id[request_id] = {
            rule.target_id: False for rule in self.rules
        }
        self.window_1_counts_by_id[request_id] = 0
        self.window_2_counts_by_id[request_id] = 0
        self.nonfinite_steps.pop(request_id, None)

    def remove(self, request_id: str) -> None:
        self.prompt_evaluated.discard(request_id)
        for store in (
            self.prompt_scores_by_id,
            self.window_1_scores_by_id,
            self.window_2_scores_by_id,
            self.fusion_scores_by_id,
            self.action_steps_by_id,
            self.actions_by_id,
            self.window_1_sums_by_id,
            self.window_2_sums_by_id,
            self.window_1_counts_by_id,
            self.window_2_counts_by_id,
            self.nonfinite_steps,
        ):
            store.pop(request_id, None)

    def needs_prompt(self, request_id: str) -> bool:
        return request_id not in self.prompt_evaluated

    def nonfinite_step_for(self, request_id: str) -> int | None:
        """First step whose pooled mean, probe score or fusion score was not finite."""

        return self.nonfinite_steps.get(request_id)

    def execution_requirements(self) -> ExecutionRequirements:
        """The prompt and both decode windows, with one evaluation point."""

        split = self.window_split_step
        return ExecutionRequirements.for_profile(
            self.profile,
            windows=(
                ObservationWindow("prompt", 0, 0),
                ObservationWindow(f"steps_1_{split}", 1, split),
                ObservationWindow(
                    f"steps_{split + 1}_{self.max_decode_step}",
                    split + 1,
                    self.max_decode_step,
                ),
            ),
            evaluation_steps=(self.evaluation_step,),
            coverage=(
                f"evaluated once, in the step that processes generated token "
                f"{self.evaluation_step}; an action replaces token "
                f"{self.evaluation_step + 1}; tokens after "
                f"{self.evaluation_step} are not evaluated (policy choice); a "
                f"request that finishes before token {self.evaluation_step} is "
                "processed reaches no evaluation point"
            ),
        )

    def scores_for(self, request_id: str) -> dict[str, dict[str, float]]:
        return {
            "prompt": dict(self.prompt_scores_by_id.get(request_id, {})),
            "steps_1_32": dict(self.window_1_scores_by_id.get(request_id, {})),
            "steps_33_64": dict(self.window_2_scores_by_id.get(request_id, {})),
            "fusion": dict(self.fusion_scores_by_id.get(request_id, {})),
        }

    def score_steps_for(self, request_id: str) -> dict[str, int]:
        return dict(self.action_steps_by_id.get(request_id, {}))

    def actions_for(self, request_id: str) -> dict[str, bool]:
        return dict(self.actions_by_id.get(request_id, {}))

    def feature_maxima_for(self, request_id: str) -> dict[str, dict[int, float]]:
        del request_id
        return {}

    def evaluate(
        self,
        request_ids: list[str],
        activations: torch.Tensor,
        steps: list[int],
    ) -> list[Decision]:
        endpoints = activations.detach().to(device="cpu", dtype=torch.float32).numpy()
        decisions = []
        for row, (request_id, step) in enumerate(zip(request_ids, steps, strict=True)):
            endpoint = endpoints[row]
            if step == 0:
                decisions.append(self._score_prompt(request_id, endpoint))
            elif step <= self.window_split_step:
                self._add_window_hidden(
                    request_id,
                    endpoint,
                    self.window_1_sums_by_id,
                    self.window_1_counts_by_id,
                )
                if step == self.window_split_step:
                    self._score_window_1(request_id)
                decisions.append(_allow())
            else:
                self._add_window_hidden(
                    request_id,
                    endpoint,
                    self.window_2_sums_by_id,
                    self.window_2_counts_by_id,
                )
                if step == self.evaluation_step:
                    decisions.append(self._score_window_2_and_fuse(request_id))
                else:
                    decisions.append(_allow())
        return decisions

    @staticmethod
    def _add_window_hidden(
        request_id: str,
        endpoint: np.ndarray,
        sums_by_id: dict[str, np.ndarray],
        counts_by_id: dict[str, int],
    ) -> None:
        if request_id not in sums_by_id:
            sums_by_id[request_id] = endpoint.copy()
        else:
            sums_by_id[request_id] += endpoint
        counts_by_id[request_id] += 1

    def _score_prompt(self, request_id: str, endpoint: np.ndarray) -> Decision:
        if request_id in self.prompt_evaluated:
            return _allow()
        self.prompt_evaluated.add(request_id)
        hidden = endpoint.astype(np.float64)
        for rule in self.rules:
            self.prompt_scores_by_id[request_id][rule.target_id] = float(
                rule.prompt.score(hidden)
            )
        if not _all_finite(list(self.prompt_scores_by_id[request_id].values())):
            self.nonfinite_steps.setdefault(request_id, 0)
        return _allow()

    def _score_window_1(self, request_id: str) -> None:
        if self.window_1_counts_by_id[request_id] != self.window_split_step:
            raise AssertionError(
                "LMSM-Checkpoint requires generated hidden states 1 through 32"
            )
        hidden = (
            self.window_1_sums_by_id[request_id]
            / self.window_1_counts_by_id[request_id]
        ).astype(np.float64)
        for rule in self.rules:
            self.window_1_scores_by_id[request_id][rule.target_id] = float(
                rule.steps_1_32.score(hidden)
            )
        if not _all_finite(hidden, list(self.window_1_scores_by_id[request_id].values())):
            self.nonfinite_steps.setdefault(request_id, self.window_split_step)

    def _score_window_2_and_fuse(self, request_id: str) -> Decision:
        expected_count = self.evaluation_step - self.window_split_step
        if self.window_2_counts_by_id[request_id] != expected_count:
            raise AssertionError(
                "LMSM-Checkpoint requires generated hidden states 33 through 64"
            )
        hidden = (
            self.window_2_sums_by_id[request_id]
            / self.window_2_counts_by_id[request_id]
        ).astype(np.float64)
        triggered: list[tuple[CheckpointRule, np.ndarray, float]] = []
        for rule in self.rules:
            window_2_score = float(rule.steps_33_64.score(hidden))
            self.window_2_scores_by_id[request_id][rule.target_id] = window_2_score
            components = np.asarray(
                [
                    self.prompt_scores_by_id[request_id][rule.target_id],
                    self.window_1_scores_by_id[request_id][rule.target_id],
                    window_2_score,
                ],
                dtype=np.float64,
            )
            fusion_score = float(rule.fusion.score(components))
            self.fusion_scores_by_id[request_id][rule.target_id] = fusion_score
            crossed = fusion_score >= rule.fusion.threshold
            self.actions_by_id[request_id][rule.target_id] = bool(crossed)
            if crossed:
                self.action_steps_by_id[request_id][rule.target_id] = (
                    self.evaluation_step
                )
                triggered.append((rule, components, fusion_score))
        if not _all_finite(
            hidden,
            list(self.window_2_scores_by_id[request_id].values()),
            list(self.fusion_scores_by_id[request_id].values()),
        ):
            self.nonfinite_steps.setdefault(request_id, self.evaluation_step)

        if not triggered or not self.actions_enabled:
            return _allow()
        rule, components, fusion_score = _selected_trigger(self.profile, triggered)
        decision = _rule_decision(
            self.profile,
            rule.target_id,
            fusion_score,
            float(rule.fusion.threshold),
            "checkpoint_fusion",
        )
        decision.params.update(
            {
                "prompt_score": float(components[0]),
                "steps_1_32_score": float(components[1]),
                "steps_33_64_score": float(components[2]),
            }
        )
        return decision


class SharedFeaturePolicyEvaluator:
    """Evaluate one shared SAE, transcoder, or dense-feature snapshot."""

    observes_prompt = True
    observes_decode = True
    max_decode_step = None

    def __init__(
        self, shared_backend, bound_backends, aggregator, decider, profile
    ) -> None:
        self.shared_backend = shared_backend
        self.backend_templates = list(bound_backends)
        self.aggregator = aggregator
        self.decider = decider
        self.profile = profile
        self.backends_by_id: dict[str, list[Any]] = {}
        self.feature_maxima_by_id: dict[str, dict[str, dict[int, float]]] = {}
        self.nonfinite_steps: dict[str, int] = {}

    def add(self, request_id: str) -> None:
        backends = []
        for template in self.backend_templates:
            backend = copy(template)
            backend.reset_state()
            backends.append(backend)
        self.backends_by_id[request_id] = backends
        self.feature_maxima_by_id[request_id] = {}
        self.nonfinite_steps.pop(request_id, None)

    def remove(self, request_id: str) -> None:
        self.backends_by_id.pop(request_id, None)
        self.feature_maxima_by_id.pop(request_id, None)
        self.nonfinite_steps.pop(request_id, None)

    def nonfinite_step_for(self, request_id: str) -> int | None:
        """First step whose selected feature activations were not all finite."""

        return self.nonfinite_steps.get(request_id)

    def needs_prompt(self, request_id: str) -> bool:
        del request_id
        return True

    def execution_requirements(self) -> ExecutionRequirements:
        """The prompt and every processed generated token, each evaluated."""

        return ExecutionRequirements.for_profile(
            self.profile,
            windows=(
                ObservationWindow("prompt", 0, 0),
                ObservationWindow("decode", 1, None),
            ),
            evaluation_steps=None,
            coverage=(
                "the prompt and every processed generated token until the first "
                "crossing, each evaluated; the final generated token is never "
                "processed; the prompt evaluation is always reached"
            ),
        )

    def feature_maxima_for(self, request_id: str) -> dict[str, dict[int, float]]:
        return {
            target: dict(scores)
            for target, scores in self.feature_maxima_by_id.get(
                request_id, {}
            ).items()
        }

    def scores_for(self, request_id: str) -> dict[str, float]:
        del request_id
        return {}

    def score_steps_for(self, request_id: str) -> dict[str, int]:
        del request_id
        return {}

    def actions_for(self, request_id: str) -> dict[str, bool]:
        del request_id
        return {}

    def evaluate(
        self,
        request_ids: list[str],
        activations: torch.Tensor,
        steps: list[int],
    ) -> list[Decision]:
        draft = SimpleNamespace(
            activations={self.shared_backend.activation_key: activations.unsqueeze(1)}
        )
        snapshot = self.shared_backend.snapshot(draft)
        decisions = []
        for row, request_id in enumerate(request_ids):
            signals = []
            for backend in self.backends_by_id[request_id]:
                feature_scores = snapshot.scores(backend.feature_indices, row=row)
                if not _all_finite(feature_scores):
                    self.nonfinite_steps.setdefault(request_id, int(steps[row]))
                maxima = self.feature_maxima_by_id[request_id].setdefault(
                    backend.category, {}
                )
                for feature_id, score in zip(
                    backend.feature_indices, feature_scores, strict=True
                ):
                    previous = maxima.get(feature_id)
                    if previous is None or score > previous:
                        maxima[feature_id] = score
                signals.append(backend.observe_feature_scores(feature_scores))
            risks = self.aggregator.aggregate(signals, self.profile)
            decisions.append(self.decider.decide(risks, None, self.profile))
        return decisions


def _float64_hidden(activations: torch.Tensor) -> np.ndarray:
    return (
        activations.detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .astype(np.float64)
    )


def _all_finite(*values: Any) -> bool:
    """True when every value (a number, a list, or an array) is finite."""

    return all(
        bool(np.isfinite(np.asarray(value, dtype=np.float64)).all()) for value in values
    )


def _allow() -> Decision:
    return Decision(action="allow")


def _rule_decision(
    profile: DeploymentProfile,
    target_id: str,
    score: float,
    threshold: float,
    temporal_condition: str,
) -> Decision:
    rule = profile.rule_for_target(target_id)
    action = profile.action_for_rule(rule.rule_id)
    return Decision(
        action=action,
        params={
            "rule_id": rule.rule_id,
            "target_id": target_id,
            "category": target_id,
            "score": score,
            "threshold": threshold,
            "temporal_condition": temporal_condition,
        },
        reason=f"{rule.rule_id} crossed its {profile.policy.name} threshold",
    )


def _selected_trigger(profile: DeploymentProfile, triggered: list[tuple]):
    by_rule_id = {
        profile.rule_for_target(item[0].target_id).rule_id: item
        for item in triggered
    }
    selected_rule_id = profile.policy.compose(list(by_rule_id))
    return by_rule_id[selected_rule_id]


__all__ = [
    "AnytimePolicyEvaluator",
    "CheckpointPolicyEvaluator",
    "SharedFeaturePolicyEvaluator",
]
