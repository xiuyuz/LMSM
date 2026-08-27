from __future__ import annotations

import os
from pathlib import Path


class LinearTranscoderAdapter:
    """Minimal encode/decode wrapper for Neuronpedia skip-transcoder checkpoints.

    This is sufficient for backend-only runs, where LMSM needs feature
    activations via encode(...). decode(...) is included for interface parity,
    but does not currently apply the residual skip path.
    """

    def __init__(self, *, W_enc, b_enc, W_dec, b_dec, W_skip=None):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.W_skip = W_skip

    def encode(self, x):
        import torch

        return torch.relu(x @ self.W_enc.transpose(0, 1) + self.b_enc)

    def encode_selected(self, x, feature_indices):
        """Encode only the requested transcoder coordinates."""
        import torch

        selected_W_enc = self.W_enc[feature_indices]
        selected_b_enc = self.b_enc[feature_indices]
        return torch.relu(x @ selected_W_enc.transpose(0, 1) + selected_b_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def load_sae(
    path: str | Path | None = None,
    *,
    source: str = "disk",
    release: str | None = None,
    sae_id: str | None = None,
    repo_id: str | None = None,
    filename: str | None = None,
    device: str = "cpu",
):
    if source == "hf":
        try:
            from sae_lens import SAE
        except ModuleNotFoundError as exc:
            raise RuntimeError("sae_lens is required to load SAEs from SAELens releases") from exc
        if release is None or sae_id is None:
            raise ValueError("SAELens release loading requires both release and sae_id")
        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        return loaded[0] if isinstance(loaded, (tuple, list)) else loaded

    if source == "hf_hub":
        if repo_id is None or filename is None:
            raise ValueError("hf_hub loading requires both repo_id and filename")
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:
            raise RuntimeError("huggingface_hub is required for source: hf_hub") from exc
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        path = downloaded_path

    if path is None:
        raise ValueError("Disk SAE loading requires a path")

    sae_path = Path(path)
    if sae_path.is_dir():
        try:
            from sae_lens import SAE
        except ModuleNotFoundError:
            pass
        else:
            return SAE.load_from_disk(sae_path, device=device)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to load SAE artifacts from torch checkpoints") from exc

    if sae_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("safetensors is required to load .safetensors SAE artifacts") from exc
        state_dict = load_file(str(sae_path), device=device or "cpu")
        if {"W_enc", "b_enc", "W_dec", "b_dec"}.issubset(state_dict):
            return LinearTranscoderAdapter(
                W_enc=state_dict["W_enc"],
                b_enc=state_dict["b_enc"],
                W_dec=state_dict["W_dec"],
                b_dec=state_dict["b_dec"],
                W_skip=state_dict.get("W_skip"),
            )
        raise RuntimeError(
            f"Unsupported .safetensors artifact at {sae_path}: expected transcoder keys W_enc/b_enc/W_dec/b_dec"
        )

    return torch.load(sae_path, map_location="cpu")
