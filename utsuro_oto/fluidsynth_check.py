"""Lightweight FluidSynth availability probe.

Adapted from RemiFabre/Theremini.  Exercises the shared library at startup
so we get a clear error message before entering the main loop.

Prefers the scamp bundled fluidsynth wrapper (``scamp._dependencies.fluidsynth``)
because that is the binary actually used at runtime on the Reachy Mini.
Falls back to the standalone ``pyFluidSynth`` package if the bundled copy is
unavailable (e.g. in a minimal CI environment).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FluidSynthProbeResult:
    ok: bool
    error: str | None = None


def probe_fluidsynth() -> FluidSynthProbeResult:
    """Try to initialise FluidSynth and play a silent test note.

    Prefers ``scamp._dependencies.fluidsynth`` (the bundled wrapper scamp uses
    at runtime).  Falls back to the standalone ``pyFluidSynth`` package.

    Returns a result with ``ok=True`` if a working FluidSynth runtime is found,
    or ``ok=False`` with a descriptive ``error`` string otherwise.
    """
    # Prefer the bundled copy scamp loads at runtime (verified on Reachy Mini).
    try:
        from scamp._dependencies import fluidsynth  # type: ignore[import]

        if fluidsynth is None:
            raise ImportError("scamp._dependencies.fluidsynth is None")
        source = "bundled (scamp)"
    except Exception:
        try:
            import fluidsynth  # type: ignore[import]  # noqa: PLC0415

            source = "system (pyFluidSynth)"
        except Exception as exc:
            return FluidSynthProbeResult(False, f"FluidSynth not available: {exc}")

    try:
        synth = fluidsynth.Synth()
    except Exception as exc:
        return FluidSynthProbeResult(False, f"Unable to initialise {source} FluidSynth: {exc}")

    try:
        synth.noteon(0, 60, 30)
        synth.noteoff(0, 60)
    except Exception as exc:
        try:
            synth.delete()
        except Exception:
            pass
        return FluidSynthProbeResult(False, f"{source} FluidSynth test note failed: {exc}")

    try:
        synth.delete()
    except Exception:
        pass

    return FluidSynthProbeResult(True, None)


__all__ = ["probe_fluidsynth", "FluidSynthProbeResult"]
