"""SoundEngine: FluidSynth/scamp wrapper for theremin-style continuous note playback.

Adapted from RemiFabre/Theremini.

The engine holds a single sustained note that can have its pitch and volume
changed in real-time without gaps or clicks (using scamp's ``start_note`` /
``change_note_pitch`` / ``change_note_volume`` API).

Typical usage in a 50 Hz control loop::

    engine = SoundEngine()
    if engine.ok:
        engine.play_or_update(pitch=60, amplitude=0.8, instrument="flute")
    # ... later when hand disappears
        engine.stop_note()
"""
from __future__ import annotations

import contextlib
import io
from typing import Any

from thereminvox.fluidsynth_check import FluidSynthProbeResult, probe_fluidsynth


class SoundEngine:
    """Manages a scamp Session and a single sustained theremin note."""

    def __init__(self, initial_instrument: str = "flute") -> None:
        self._probe: FluidSynthProbeResult = probe_fluidsynth()
        self._session: Any | None = None
        self._parts_cache: dict[str, Any] = {}
        self._note_handle: Any | None = None
        self._current_pitch: int | None = None
        self._current_instrument: str | None = None

        if self._probe.ok:
            try:
                from scamp import Session as ScampSession  # type: ignore[import]
            except Exception as exc:
                self._probe = FluidSynthProbeResult(False, f"scamp import failed: {exc}")
            else:
                self._session = ScampSession(max_threads=1024)
                # Pre-load the initial instrument to avoid first-note latency.
                self._get_part(initial_instrument)

    # ── Public interface ───────────────────────────────────────────

    @property
    def ok(self) -> bool:
        return self._probe.ok

    @property
    def error(self) -> str | None:
        return self._probe.error

    def play_or_update(
        self,
        pitch: int,
        amplitude: float,
        instrument: str,
    ) -> None:
        """Start or update the sustained theremin note.

        If no note is playing, starts a new one.  If the instrument has
        changed, the old note is ended and a new one started.  Otherwise
        pitch and volume are updated smoothly without retriggering.

        Args:
            pitch:      MIDI note number (48–84 typical).
            amplitude:  Volume in [0.0, 1.0].
            instrument: scamp instrument name (GM preset string).

        """
        if self._session is None:
            return

        instr = self._get_part(instrument)
        if instr is None:
            return

        if self._note_handle is None or self._current_instrument != instrument:
            # Instrument changed or no note running: start fresh.
            self._stop_note()
            self._note_handle = instr.start_note(pitch, amplitude)
            self._current_pitch = pitch
            self._current_instrument = instrument
            return

        # Same instrument, update in place.
        if pitch != self._current_pitch:
            self._note_handle.change_pitch(pitch)
            self._current_pitch = pitch
        self._note_handle.change_volume(amplitude)

    def stop_note(self) -> None:
        """Silence the current note (call when hand disappears or on shutdown)."""
        self._stop_note()

    def shutdown(self) -> None:
        """End the note and clean up resources."""
        self._stop_note()

    # ── Internal helpers ───────────────────────────────────────────

    def _get_part(self, name: str) -> Any | None:
        if self._session is None:
            return None
        if name not in self._parts_cache:
            # scamp prints info on part creation; suppress it.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self._parts_cache[name] = self._session.new_part(name)
        return self._parts_cache[name]

    def _stop_note(self) -> None:
        if self._note_handle is not None:
            try:
                self._note_handle.end()
            except Exception:
                pass
            self._note_handle = None
            self._current_pitch = None
