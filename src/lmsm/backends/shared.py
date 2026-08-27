from __future__ import annotations

from dataclasses import dataclass

from lmsm.backends.sae import SAEBackend
from lmsm.utils.values import last_token_representation, to_row_vector


@dataclass
class FeatureSnapshot:
    """The compact, sorted feature view produced by one shared backend."""

    feature_indices: tuple[int, ...]
    values: object

    def scores(self, feature_indices: list[int], *, row: int = 0) -> list[float]:
        positions = [self.feature_indices.index(index) for index in feature_indices]
        if hasattr(self.values, "dim"):
            values = self.values if self.values.dim() == 1 else self.values[row]
            return [float(values[position].item()) for position in positions]

        values = self.values
        if values and isinstance(values[0], (list, tuple)):
            values = values[row]
        return [float(values[position]) for position in positions]


class SharedFeatureBackend:
    """Compute one artifact/site encoding shared by all bound rule backends."""

    def __init__(self, artifact, activation_key: str, feature_indices, *, selected: bool):
        self.artifact = artifact
        self.activation_key = activation_key
        self.feature_indices = tuple(sorted(set(feature_indices)))
        self.selected = bool(selected)
        self.encode_count = 0

    def snapshot(self, draft) -> FeatureSnapshot:
        activation = (draft.activations or {}).get(self.activation_key)
        if activation is None:
            raise KeyError(f"Missing activation for {self.activation_key}")

        token_representation = to_row_vector(last_token_representation(activation))
        if self.selected and hasattr(self.artifact, "encode_selected"):
            encoded = self.artifact.encode_selected(
                token_representation,
                list(self.feature_indices),
            )
        else:
            full = (
                self.artifact.encode(token_representation)
                if hasattr(self.artifact, "encode")
                else self.artifact(token_representation)
            )
            encoded = _select_features(full, self.feature_indices)
        self.encode_count += 1
        return FeatureSnapshot(self.feature_indices, encoded)


class BoundFeatureBackend(SAEBackend):
    """One rule backend bound to positions in a shared feature snapshot."""

    def __init__(self, *, shared_backend: SharedFeatureBackend, **kwargs):
        super().__init__(
            sae=shared_backend.artifact,
            activation_key=shared_backend.activation_key,
            **kwargs,
        )
        self.shared_backend = shared_backend

    def observe_snapshot(self, snapshot: FeatureSnapshot):
        return self.observe_feature_scores(snapshot.scores(self.feature_indices))


def _select_features(encoded, feature_indices: tuple[int, ...]):
    if hasattr(encoded, "dim"):
        return encoded[..., list(feature_indices)]
    if encoded and isinstance(encoded[0], (list, tuple)):
        return [[row[index] for index in feature_indices] for row in encoded]
    return [encoded[index] for index in feature_indices]
