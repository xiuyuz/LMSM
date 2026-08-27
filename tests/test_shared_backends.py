from types import SimpleNamespace

import torch

from lmsm.backends.shared import BoundFeatureBackend, SharedFeatureBackend
from lmsm.loaders.sae_loader import LinearTranscoderAdapter


class CountingAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.full_calls = 0
        self.selected_calls = 0

    def encode(self, hidden):
        self.full_calls += 1
        return self.adapter.encode(hidden)

    def encode_selected(self, hidden, feature_indices):
        self.selected_calls += 1
        return self.adapter.encode_selected(hidden, feature_indices)


def _adapter():
    return LinearTranscoderAdapter(
        W_enc=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 0.0],
                [-1.0, 0.0, 2.0],
            ]
        ),
        b_enc=torch.tensor([0.1, -0.2, 0.3, 0.0, 0.4]),
        W_dec=torch.zeros(5, 3),
        b_dec=torch.zeros(3),
    )


def test_selected_features_match_full_encoding_and_are_shared_across_rules():
    counting = CountingAdapter(_adapter())
    selected_backend = SharedFeatureBackend(
        counting, "site", [4, 1, 3, 1], selected=True
    )
    full_backend = SharedFeatureBackend(
        _adapter(), "site", [4, 1, 3], selected=False
    )
    draft = SimpleNamespace(
        activations={"site": torch.tensor([[[0.5, 1.5, -0.25]]])}
    )

    selected = selected_backend.snapshot(draft)
    full = full_backend.snapshot(draft)
    torch.testing.assert_close(selected.values, full.values)
    assert selected.feature_indices == (1, 3, 4)
    assert counting.selected_calls == 1
    assert counting.full_calls == 0

    first = BoundFeatureBackend(
        shared_backend=selected_backend,
        name="chemical_biological",
        category="chemical_biological",
        feature_indices=[4, 1],
        threshold=1.0,
    )
    second = BoundFeatureBackend(
        shared_backend=selected_backend,
        name="copyright",
        category="copyright",
        feature_indices=[3, 1],
        threshold=1.0,
    )
    first_signal = first.observe_snapshot(selected)
    second_signal = second.observe_snapshot(selected)

    assert first_signal.backend == "chemical_biological"
    assert second_signal.backend == "copyright"
    assert first_signal.info["feature_indices"] == [4, 1]
    assert second_signal.info["feature_indices"] == [3, 1]
    assert counting.selected_calls == 1
    assert selected_backend.encode_count == 1
