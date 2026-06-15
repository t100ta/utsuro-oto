"""Lightweight FluidSynth availability probe.

Adapted from RemiFabre/Theremini.  Exercises the shared library at startup
so we get a clear error message before entering the main loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FluidSynthProbeResult:
    ok: bool
    error: str | None = None


def probe_fluidsynth() -> FluidSynthProbeResult:
    """Try to initialise FluidSynth and play a silent test note.

    Returns a result with ``ok=True`` if the runtime library is available,
    or ``ok=False`` with a descriptive ``error`` string otherwise.
    """
    try:
        import fluidsynth  # type: ignore[import]
    except Exception as exc:
        return FluidSynthProbeResult(False, f"pyFluidSynth import failed: {exc}")

    try:
        synth = fluidsynth.Synth()
    except Exception as exc:
        return FluidSynthProbeResult(False, f"Unable to initialise FluidSynth: {exc}")

    try:
        synth.noteon(0, 60, 30)
        synth.noteoff(0, 60)
    except Exception as exc:
        synth.delete()
        return FluidSynthProbeResult(False, f"FluidSynth test note failed: {exc}")

    try:
        synth.delete()
    except Exception:
        pass

    return FluidSynthProbeResult(True, None)


__all__ = ["probe_fluidsynth", "FluidSynthProbeResult"]
