"""SAE interpretability backend for sparse-autoencoder feature signals."""
from __future__ import annotations

from lmsm.backends.base import BaseBackend
from lmsm.state import BackendSignal
from lmsm.utils.values import (
    coerce_float,
    last_token_representation,
    take_feature,
    to_row_vector,
)


class SAEBackend(BaseBackend):
    """Produce safety signals from SAE feature activations.

    Score in [0, 1]: fraction of watched features that exceed their thresholds.
    A single-feature backend returns 0.0 or 1.0 (binary: fired / did not fire).
    """

    def __init__(
        self,
        name: str,
        category: str,
        sae,
        activation_key: str,
        feature_indices: list[int],
        threshold: float,
        feature_thresholds: list[float] | None = None,
        feature_group_rule: str = "any",
        min_feature_fraction: float = 1.0,
    ):
        self.name = name
        self.category = category
        self.sae = sae
        self.activation_key = activation_key
        self.feature_indices = list(feature_indices)
        self.threshold = float(threshold)
        self.feature_thresholds = _resolve_feature_thresholds(
            self.feature_indices,
            self.threshold,
            feature_thresholds,
        )
        self.feature_group_rule = feature_group_rule
        self.min_feature_fraction = float(min_feature_fraction)
        self.running_max: dict[int, float] = {}

    def reset_state(self) -> None:
        self.running_max = {}

    def observe(self, draft) -> BackendSignal:
        activation = (draft.activations or {}).get(self.activation_key)
        if activation is None:
            raise KeyError(f"Missing activation for {self.activation_key}")

        token_representation = to_row_vector(last_token_representation(activation))
        encoded = (
            self.sae.encode(token_representation)
            if hasattr(self.sae, "encode")
            else self.sae(token_representation)
        )
        feature_scores = [
            coerce_float(take_feature(encoded, feature_idx))
            for feature_idx in self.feature_indices
        ]

        return self.observe_feature_scores(feature_scores)

    def observe_feature_scores(self, feature_scores: list[float]) -> BackendSignal:
        """Apply backend state and thresholds to already-encoded features."""

        for fi, score in zip(self.feature_indices, feature_scores):
            if score > self.running_max.get(fi, 0.0):
                self.running_max[fi] = score

        feature_passes = [
            feature_score >= feature_threshold
            for feature_score, feature_threshold in zip(
                feature_scores, self.feature_thresholds
            )
        ]
        pass_count = sum(1 for passed in feature_passes if passed)
        pass_fraction = pass_count / max(len(self.feature_indices), 1)

        cumulative_passes = [
            self.running_max.get(fi, 0.0) >= thr
            for fi, thr in zip(self.feature_indices, self.feature_thresholds)
        ]
        cumulative_pass_count = sum(1 for p in cumulative_passes if p)
        cumulative_pass_fraction = cumulative_pass_count / max(len(self.feature_indices), 1)

        triggered = _group_is_triggered(
            feature_passes,
            cumulative_passes=cumulative_passes,
            feature_group_rule=self.feature_group_rule,
            min_feature_fraction=self.min_feature_fraction,
        )
        return BackendSignal(
            backend=self.name,
            category=self.category,
            score=(
                cumulative_pass_fraction
                if self.feature_group_rule == "cumulative_fraction"
                else pass_fraction
            ),
            triggered=triggered,
            info={
                "activation_key": self.activation_key,
                "feature_indices": list(self.feature_indices),
                "feature_scores": feature_scores,
                "feature_thresholds": list(self.feature_thresholds),
                "feature_passes": feature_passes,
                "pass_count": pass_count,
                "pass_fraction": pass_fraction,
                "cumulative_passes": cumulative_passes,
                "cumulative_pass_count": cumulative_pass_count,
                "cumulative_pass_fraction": cumulative_pass_fraction,
                "feature_group_rule": self.feature_group_rule,
                "min_feature_fraction": self.min_feature_fraction,
            },
        )


def _resolve_feature_thresholds(
    feature_indices: list[int],
    threshold: float,
    feature_thresholds: list[float] | None,
) -> list[float]:
    if feature_thresholds is None:
        return [float(threshold) for _ in feature_indices]
    if len(feature_thresholds) != len(feature_indices):
        raise ValueError("feature_thresholds must match feature_indices in length")
    return [float(value) for value in feature_thresholds]


def _group_is_triggered(
    feature_passes: list[bool],
    *,
    cumulative_passes: list[bool],
    feature_group_rule: str,
    min_feature_fraction: float,
) -> bool:
    if not feature_passes:
        return False
    if feature_group_rule == "any":
        return any(feature_passes)
    if feature_group_rule == "fraction":
        pass_fraction = sum(1 for passed in feature_passes if passed) / len(feature_passes)
        return pass_fraction >= min_feature_fraction
    if feature_group_rule == "cumulative_fraction":
        cum_fraction = sum(1 for passed in cumulative_passes if passed) / len(cumulative_passes)
        return cum_fraction >= min_feature_fraction
    raise ValueError(f"Unknown feature_group_rule: {feature_group_rule}")
