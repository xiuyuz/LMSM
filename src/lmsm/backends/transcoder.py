"""Transcoder interpretability backend.

An SAE reconstructs its input, such as the residual stream.  A transcoder, such
as mwhanna/qwen3-4b-transcoders, predicts an MLP's output from the MLP's input.
Both expose the same linear encode, so TranscoderBackend is a named subclass of
SAEBackend.  The separate name shows in logs and profiles which kind of
artifact a backend uses.
"""
from __future__ import annotations

from lmsm.backends.sae import SAEBackend


class TranscoderBackend(SAEBackend):
    """Produce safety signals from transcoder feature activations.

    The logic is SAEBackend's.  The separate name shows in engine state and
    logs which kind of artifact is in use.
    """
