"""Shared helper: torch device resolution."""

from __future__ import annotations


def resolve_device(preference: str = "auto") -> str:
    """Resolve a torch device string.

    ``"auto"`` picks cuda → mps → cpu based on availability. An explicit value
    (``"cpu"``, ``"mps"``, ``"cuda"``) is returned unchanged.
    """
    # Imported in-function, not at module scope: torch is in the `ml` dependency group,
    # and this module is imported by code that runs without that group installed.
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
