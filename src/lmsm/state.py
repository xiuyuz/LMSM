from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepState:
    step_idx: int
    input_ids: Any
    generated_ids: Any
    cache: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDraft:
    step_idx: int
    input_ids: Any
    generated_ids: Any
    logits: Any
    next_token_id: Any
    activations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendSignal:
    backend: str
    category: str
    score: float
    triggered: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    category: str
    score: float
    level: str
    triggered: bool
    supporting_backends: list[str]
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
