"""SoundEngine: FluidSynth/scamp wrapper for theremin-style continuous note playback.

Adapted from RemiFabre/Theremini.

The engine holds a single sustained note that can have its pitch and volume
changed in real-time without gaps or clicks (using scamp's ``start_note`` /
``change_pitch`` / ``change_volume`` API).

Typical usage in a 50 Hz control loop::

    engine = SoundEngine()
    if engine.ok:
        engine.play_or_update(pitch=60, amplitude=0.8, instrument="flute")
    # ... later when hand disappears
        engine.stop_note()

Audio-routing environment variables (set before launching the app)::

    THEREMINVOX_AUDIO_DRIVER   scamp/fluidsynth driver (default: "auto").
                               On the robot: try "alsa".
    THEREMINVOX_AUDIO_DEVICE   ALSA device name (default: unset).
                               On the robot: try "plug:reachymini_audio_sink".
    THEREMINVOX_SAMPLE_RATE    Override synth sample rate in Hz (default: unset).
                               On the robot: try "16000" to match the dmix sink.
    THEREMINVOX_SOUNDFONT      scamp soundfont name or path (default: "default" = Merlin.sf2).
    THEREMINVOX_AUDIO_TEST     Set to "0" to skip the 1.5 s startup self-test tone (default: "1").
"""
from __future__ import annotations

import contextlib
import io
import os
import time
from typing import Any

from thereminvox.fluidsynth_check import FluidSynthProbeResult, probe_fluidsynth

# ── Environment-variable configuration ──────────────────────────────────────
_AUDIO_DRIVER = os.environ.get("THEREMINVOX_AUDIO_DRIVER", "default")  # "default" = let scamp auto-detect
_SOUNDFONT    = os.environ.get("THEREMINVOX_SOUNDFONT",    "default")
_AUDIO_DEVICE = os.environ.get("THEREMINVOX_AUDIO_DEVICE", "")
_SAMPLE_RATE  = os.environ.get("THEREMINVOX_SAMPLE_RATE",  "")
_AUDIO_TEST   = os.environ.get("THEREMINVOX_AUDIO_TEST",   "1") != "0"


def _patch_fluidsynth_start() -> None:
    """Inject THEREMINVOX_AUDIO_DEVICE / THEREMINVOX_SAMPLE_RATE into fluidsynth.Synth.start.

    scamp calls ``self.synth.start(driver=...)`` but never forwards ``device`` or
    ``sample-rate``.  pyfluidsynth already supports both — scamp just doesn't
    expose them.  This patch is a strict no-op when neither env var is set.
    """
    if not (_AUDIO_DEVICE or _SAMPLE_RATE):
        return
    try:
        import fluidsynth  # type: ignore[import]
        _orig = fluidsynth.Synth.start

        def _patched(synth_self: Any, driver: str = "alsa", device: str | None = None, **kw: Any) -> Any:
            if _SAMPLE_RATE:
                try:
                    synth_self.setting("synth.sample-rate", float(_SAMPLE_RATE))
                except Exception:
                    pass
            effective_device = device or _AUDIO_DEVICE or None
            return _orig(synth_self, driver=driver, device=effective_device, **kw)

        fluidsynth.Synth.start = _patched  # type: ignore[method-assign]
        print(f"[Audio] fluidsynth patched: device={_AUDIO_DEVICE!r} sample_rate={_SAMPLE_RATE!r}")
    except Exception:
        pass  # fluidsynth unavailable; probe_fluidsynth() will surface the real error


_patch_fluidsynth_start()


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
                print(f"[Audio] Starting scamp (driver={_AUDIO_DRIVER!r}, soundfont={_SOUNDFONT!r})")
                self._session = ScampSession(
                    max_threads=1024,
                    default_audio_driver=_AUDIO_DRIVER,
                    default_soundfont=_SOUNDFONT,
                )
                # Pre-load the initial instrument to avoid first-note latency.
                self._get_part(initial_instrument)

    # ── Public interface ───────────────────────────────────────────

    @property
    def ok(self) -> bool:
        return self._probe.ok

    @property
    def error(self) -> str | None:
        return self._probe.error

    def self_test(self) -> None:
        """Play a 1.5 s tone (C4) at startup to confirm audio routing works.

        Disabled by setting ``THEREMINVOX_AUDIO_TEST=0``.
        """
        if not self.ok or not _AUDIO_TEST:
            return
        part = self._get_part(self._current_instrument or "flute")
        if part is None:
            return
        print("[Audio] self-test: playing 1.5 s tone (C4) … (THEREMINVOX_AUDIO_TEST=0 to skip)")
        try:
            handle = part.start_note(60, 0.8)
            time.sleep(1.5)
            handle.end()
            print("[Audio] self-test: done.")
        except Exception as exc:
            print(f"[Audio] self-test failed: {exc}")

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
            # Capture scamp's part-creation output and print it so that
            # soundfont / preset warnings are visible rather than silently dropped.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self._parts_cache[name] = self._session.new_part(name)
            msg = buf.getvalue().strip()
            if msg:
                print(f"[scamp] {msg}")
        return self._parts_cache[name]

    def _stop_note(self) -> None:
        if self._note_handle is not None:
            try:
                self._note_handle.end()
            except Exception:
                pass
            self._note_handle = None
            self._current_pitch = None
