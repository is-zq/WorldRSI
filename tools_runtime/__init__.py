"""Local solver and renderer tools for a single WorldRSI task.

No agent scheduling, model calls, or video quality acceptance runs on import.
"""
from ._vendor import depth, godot, media, physics, proxy, simulators
from .runtime import (
    prepare_first_frame,
    render,
    restore,
    scaffold,
    simulate,
    validate_render,
)

__all__ = [
    "depth", "godot", "media", "physics", "proxy", "simulators",
    "prepare_first_frame", "render", "restore", "scaffold", "simulate",
    "validate_render",
]
