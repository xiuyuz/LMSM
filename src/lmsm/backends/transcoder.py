"""Transcoder interpretability backend.

Transcoders (e.g. mwhanna/qwen3-4b-transcoders) differ from SAEs conceptually:
- SAE: reconstructs the residual stream (input approx decode(encode(input)))
- Transcoder: predicts MLP output from MLP input (output approx decode(encode(mlp_input)))

Both expose the same linear encode interface, so TranscoderBackend is a thin
named subclass of SAEBackend. Having a distinct class name keeps the "SAE
backend" and "transcoder backend" visibly separate in logs and configuration
about backend modularity.
"""
from __future__ import annotations

from lmsm.backends.sae import SAEBackend


class TranscoderBackend(SAEBackend):
    """Produce safety signals from transcoder feature activations.

    Identical logic to SAEBackend; the distinct name clarifies which artifact
    type is in use when inspecting engine state or reading logs.
    """
