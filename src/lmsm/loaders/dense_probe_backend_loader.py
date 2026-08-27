from __future__ import annotations

import json
from pathlib import Path


class DenseProbeBackend:
    """Expose fitted standardized dense probes through the backend interface."""

    def __init__(self, directions, biases):
        self.directions = directions
        self.biases = biases

    def encode(self, x):
        return x.float() @ self.directions.transpose(0, 1) + self.biases

    def encode_selected(self, x, feature_indices):
        indices = list(feature_indices)
        return x.float() @ self.directions[indices].transpose(0, 1) + self.biases[indices]


def load_dense_probe_backend(
    report_path: str | Path,
    categories: list[str],
    *,
    device: str = "cpu",
) -> DenseProbeBackend:
    """Load fitted standardized logistic probes as scalar backend outputs."""

    import torch

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    fit = report["full_development_fit"]
    if fit.get("available") is not True:
        raise ValueError("dense probe fit is unavailable")

    directions = []
    biases = []
    for category in categories:
        rule = fit["rules"][category]
        if rule.get("convergence") != "ok":
            raise ValueError(f"dense probe {category!r} did not converge")
        mean = torch.tensor(rule["scaler"]["mean"], dtype=torch.float32)
        scale = torch.tensor(rule["scaler"]["scale"], dtype=torch.float32)
        coefficient = torch.tensor(rule["classifier"]["coef"], dtype=torch.float32)
        direction = coefficient / scale
        directions.append(direction)
        biases.append(float(rule["classifier"]["intercept"]) - float(mean @ direction))

    return DenseProbeBackend(
        torch.stack(directions).to(device),
        torch.tensor(biases, dtype=torch.float32, device=device),
    )
