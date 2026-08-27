"""Task-fitted dense probes used by LMSM policy evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DenseProbe:
    """A standardized linear probe with its calibrated action threshold."""

    target_id: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.float64
    threshold: np.float64

    def score(self, hidden_float64: np.ndarray) -> np.ndarray:
        return (
            (hidden_float64 - self.mean) / self.scale
        ) @ self.coef + self.intercept


@dataclass
class CheckpointRule:
    """The four calibrated probes evaluated by an LMSM-Checkpoint rule."""

    target_id: str
    prompt: DenseProbe
    steps_1_32: DenseProbe
    steps_33_64: DenseProbe
    fusion: DenseProbe


def load_anytime_probes(
    parameters: dict[str, Any], target_ids: list[str]
) -> list[DenseProbe]:
    """Load Anytime probes in the caller's target order."""

    return _load_probes(parameters, target_ids)


def load_checkpoint_rules(
    parameters: dict[str, Any], target_ids: list[str]
) -> list[CheckpointRule]:
    """Load Checkpoint rules in the caller's target order."""

    rules = []
    for target_id in target_ids:
        values = parameters["rules"][target_id]
        threshold = np.float64(values["threshold"])
        rules.append(
            CheckpointRule(
                target_id=target_id,
                prompt=_load_probe(
                    target_id,
                    values["prompt"],
                    threshold=np.float64(0.0),
                ),
                steps_1_32=_load_probe(
                    target_id,
                    values["steps_1_32"],
                    threshold=np.float64(0.0),
                ),
                steps_33_64=_load_probe(
                    target_id,
                    values["steps_33_64"],
                    threshold=np.float64(0.0),
                ),
                fusion=_load_probe(
                    target_id,
                    values["fusion"],
                    threshold=threshold,
                ),
            )
        )
    return rules


def _load_probes(
    parameters: dict[str, Any], target_ids: list[str]
) -> list[DenseProbe]:
    probes = []
    for target_id in target_ids:
        values = parameters["probes"][target_id]
        probes.append(
            _load_probe(
                target_id,
                values,
                threshold=np.float64(values["threshold"]),
            )
        )
    return probes


def _load_probe(
    target_id: str,
    values: dict[str, Any],
    *,
    threshold: np.float64,
) -> DenseProbe:
    mean = np.asarray(values["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(values["scaler"]["scale"], dtype=np.float64)
    coef = np.asarray(values["classifier"]["coef"], dtype=np.float64)
    intercept = np.float64(values["classifier"]["intercept"])
    return DenseProbe(
        target_id=target_id,
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
        threshold=np.float64(threshold),
    )


__all__ = [
    "CheckpointRule",
    "DenseProbe",
    "load_anytime_probes",
    "load_checkpoint_rules",
]
