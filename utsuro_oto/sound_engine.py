"""SoundEngine: FluidSynth offline renderer + Reachy Mini SDK media pump.

The engine renders MIDI via FluidSynth in *offline* mode (no audio hardware
driver is opened) at 16 kHz and pushes float32 PCM through
``reachy_mini.media.push_audio_sample()``.

This reuses the same GStreamer audio path the Reachy Mini's built-in emotes
and ``play_sound()`` use (appsrc → reachymini_audio_sink), which is the only
route proven to reach the physical speaker on the robot.

Typical usage::

    engine = SoundEngine()
    engine.attach_media(reachy_mini.media)    # wire up SDK output + start pump
    engine.self_test()                        # optional 1.5 s C4 tone at startup
    # In the 50 Hz control loop:
    engine.play_or_update(pitch=60, amplitude=0.8, instrument="flute")
    # When hand disappears:
    engine.stop_note()
    # On shutdown:
    engine.shutdown()

Environment variables::

    UTSURO_OTO_SOUNDFONT   Soundfont name or path (default: "default" = Merlin.sf2
                            bundled with scamp).
    UTSURO_OTO_GAIN        FluidSynth master gain (default: "0.5"; FluidSynth default
                            is 0.2 which is quiet — increase to 1.0 if still too soft).
    UTSURO_OTO_AUDIO_TEST  Set to "0" to skip the 1.5 s startup self-test tone
                            (default: enabled).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np

from utsuro_oto.fluidsynth_check import FluidSynthProbeResult, probe_fluidsynth

# ── Bundled fluidsynth (scamp._thirdparty) + soundfont helpers ───────────────
# scamp 0.9.5 ships its own pyfluidsynth wrapper and libfluidsynth binary
# (playback_settings.use_bundled_pyfluidsynth=True by default). We import from
# there to use the exact binary that loads successfully on the Reachy Mini.
# We deliberately never call Synth.start(), which would open a hardware audio
# driver. Instead we call get_samples() for offline PCM rendering and push the
# result to the SDK's media pipeline.
try:
    from scamp._dependencies import fluidsynth  # type: ignore[import]
    from scamp._soundfont_host import (  # type: ignore[import]
        get_best_preset_match_for_name,
        resolve_soundfont,
    )
except Exception:
    fluidsynth = None  # type: ignore[assignment]
    resolve_soundfont = None  # type: ignore[assignment]
    get_best_preset_match_for_name = None  # type: ignore[assignment]

# ── Environment-variable configuration ──────────────────────────────────────
_SOUNDFONT  = os.environ.get("UTSURO_OTO_SOUNDFONT",  "default")
_GAIN       = float(os.environ.get("UTSURO_OTO_GAIN", "0.5"))   # FluidSynth default 0.2 is too quiet
_AUDIO_TEST = os.environ.get("UTSURO_OTO_AUDIO_TEST", "1") != "0"

# Duration of the startup self-test tone in seconds.
# Set to 0.0 in unit tests to skip the sleep without patching the time module.
_SELF_TEST_DURATION: float = 1.5

# When _SOUNDFONT is "default", map to scamp's "general_midi" named soundfont
# (→ Merlin.sf2, bundled with scamp at scamp/soundfonts/Merlin.sf2).
_SOUNDFONT_KEY = "general_midi" if _SOUNDFONT == "default" else _SOUNDFONT

# ── PCM pump constants ───────────────────────────────────────────────────────
OUTPUT_RATE     = 16_000   # Hz — matches reachy_mini AudioBase.SAMPLE_RATE
OUTPUT_CHANNELS = 2        # stereo — matches reachy_mini AudioBase.CHANNELS
PUMP_CHUNK      = 1024     # FluidSynth samples per pump iteration
CHAN            = 0        # FluidSynth MIDI channel used for the theremin voice


class SoundEngine:
    """FluidSynth offline renderer pushed through the Reachy Mini SDK media pipeline.

    The synth is initialised at 16 kHz but ``Synth.start()`` is never called, so
    no hardware audio driver is opened.  A background pump thread calls
    ``get_samples()`` and forwards the PCM to ``reachy_mini.media.push_audio_sample()``.
    """

    def __init__(self, initial_instrument: str = "flute") -> None:
        self._probe: FluidSynthProbeResult = probe_fluidsynth()
        self._synth: Any | None = None
        self._sfid: Any | None = None
        self._preset_cache: dict[str, tuple[int, int]] = {}
        self._synth_lock = threading.Lock()
        self._current_pitch: int | None = None
        self._current_instrument: str | None = None
        self._media: Any | None = None
        self._running = False
        self._pump_thread: threading.Thread | None = None

        if not self._probe.ok:
            return

        if fluidsynth is None or resolve_soundfont is None:
            self._probe = FluidSynthProbeResult(False, "scamp bundled fluidsynth not importable")
            return

        try:
            # Offline render only — start() is intentionally not called.
            self._synth = fluidsynth.Synth(samplerate=OUTPUT_RATE, gain=_GAIN)
            soundfont_path = resolve_soundfont(_SOUNDFONT_KEY)
            self._sfid = self._synth.sfload(soundfont_path)
            print(f"[Audio] FluidSynth ready — soundfont: {soundfont_path!r} gain={_GAIN}")
            # Pre-load the initial instrument to avoid first-note latency.
            self._apply_preset(initial_instrument)
        except Exception as exc:
            self._probe = FluidSynthProbeResult(False, f"FluidSynth init failed: {exc}")
            if self._synth is not None:
                try:
                    self._synth.delete()
                except Exception:
                    pass
                self._synth = None

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def ok(self) -> bool:
        return self._probe.ok

    @property
    def error(self) -> str | None:
        return self._probe.error

    def attach_media(self, media: Any) -> None:
        """Wire up the Reachy Mini media manager and start the PCM pump thread.

        Must be called before ``play_or_update`` will produce audible output.
        If ``media.audio`` is not available, audio is silently disabled but head
        tracking continues unaffected.

        Args:
            media: A ``reachy_mini.MediaManager`` (or any object with
                   ``start_playing()`` / ``push_audio_sample()`` / ``stop_playing()``
                   and an ``.audio`` sub-object).

        """
        if not self.ok or self._synth is None:
            return
        if media is None or getattr(media, "audio", None) is None:
            print("[Audio] WARNING: media.audio not available — sound disabled.")
            return

        # Sanity-check output sample rate (always 16 000 Hz on Reachy Mini).
        try:
            rate = media.audio.get_output_audio_samplerate()
            if rate != OUTPUT_RATE:
                print(
                    f"[Audio] WARNING: media output rate {rate} Hz ≠ {OUTPUT_RATE} Hz "
                    "— audio may be pitched wrong."
                )
        except Exception:
            pass

        # start_playing() initialises the GStreamer appsrc pipeline.
        # We do NOT call set_max_output_buffers() — its SDK implementation
        # hard-codes leaky=drop_old which silently discards queued audio and
        # causes dropouts. The monotonic-clock pump loop keeps the queue bounded
        # without needing to drop buffers.
        media.start_playing()
        self._media = media
        print("[Audio] media.start_playing() OK — PCM pump starting …")
        self._running = True
        self._pump_thread = threading.Thread(
            target=self._pump_loop, daemon=True, name="AudioPump"
        )
        self._pump_thread.start()

    def self_test(self) -> None:
        """Play a short tone (C4) at startup to confirm audio routing works.

        Must be called *after* ``attach_media``.  Disabled by setting
        ``UTSURO_OTO_AUDIO_TEST=0``.
        """
        if not self.ok or not _AUDIO_TEST or self._media is None:
            return
        print(
            f"[Audio] self-test: playing {_SELF_TEST_DURATION:.1f} s tone (C4) "
            "… (UTSURO_OTO_AUDIO_TEST=0 to skip)"
        )
        try:
            self.play_or_update(60, 0.8, self._current_instrument or "flute")
            time.sleep(_SELF_TEST_DURATION)
            self.stop_note()
            print("[Audio] self-test: done.")
        except Exception as exc:
            print(f"[Audio] self-test failed: {exc}")

    def play_or_update(self, pitch: int, amplitude: float, instrument: str) -> None:
        """Start or update the sustained theremin note.

        If no note is playing, starts a new one.  If the instrument has changed,
        the old note is ended and a new one started with the new preset.
        Otherwise pitch and volume are updated without retriggering.

        Safe to call when not attached (silently a no-op before ``attach_media``).

        Args:
            pitch:      MIDI note number (48–84 typical).
            amplitude:  Volume in [0.0, 1.0] (mapped to CC 11 expression).
            instrument: Instrument name string (GM preset, e.g. ``"flute"``).

        """
        if self._synth is None:
            return
        with self._synth_lock:
            self._play_or_update_locked(pitch, amplitude, instrument)

    def stop_note(self) -> None:
        """Silence the current note (call when hand disappears)."""
        if self._synth is None:
            return
        with self._synth_lock:
            self._stop_note_locked()

    def shutdown(self) -> None:
        """Stop the PCM pump, release the media pipeline, and delete the synth."""
        self._running = False
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=2.0)
            self._pump_thread = None

        with self._synth_lock:
            self._stop_note_locked()

        if self._media is not None:
            try:
                self._media.stop_playing()
            except Exception:
                pass
            self._media = None

        if self._synth is not None:
            try:
                self._synth.delete()
            except Exception:
                pass
            self._synth = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pump_loop(self) -> None:
        """Background thread: pull rendered PCM from FluidSynth → push to SDK.

        Uses a monotonic-clock compensating loop so that processing jitter does
        not accumulate into audio drift or burst-pushing.  Each iteration waits
        only for the remaining time in the current chunk window, keeping the push
        rate at exactly OUTPUT_RATE samples/second on average.
        """
        chunk_s = PUMP_CHUNK / OUTPUT_RATE
        next_t = time.monotonic() + chunk_s
        while self._running:
            if self._synth is None or self._media is None:
                time.sleep(chunk_s)
                next_t = time.monotonic() + chunk_s
                continue
            try:
                with self._synth_lock:
                    # int16 interleaved stereo, shape (2 * PUMP_CHUNK,)
                    s = self._synth.get_samples(PUMP_CHUNK)
                # Convert to float32 (N, 2) as expected by push_audio_sample.
                pcm = (s.astype(np.float32) / 32768.0).reshape(-1, 2)
                self._media.push_audio_sample(pcm)
            except Exception:
                pass  # absorb transient errors (shutdown races, etc.)
            next_t += chunk_s
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -chunk_s:
                # Fell more than one chunk behind; re-anchor to avoid burst-pushing.
                next_t = time.monotonic() + chunk_s

    def _resolve_preset(self, name: str) -> tuple[int, int] | None:
        """Return ``(bank, preset)`` for the given instrument name (cached).

        Uses scamp's fuzzy-matching ``get_best_preset_match_for_name`` so that
        instrument names like ``"flute"`` resolve to ``"Flute Gold"`` in Merlin.sf2
        — the same mapping scamp's ``Session.new_part()`` would use.
        """
        if name in self._preset_cache:
            return self._preset_cache[name]
        if get_best_preset_match_for_name is None:
            return None
        try:
            match, _ = get_best_preset_match_for_name(name, which_soundfont=_SOUNDFONT_KEY)
        except Exception as exc:
            print(f"[Audio] Preset lookup failed for {name!r}: {exc}")
            return None
        if match is None:
            return None
        print(f"[scamp] Using preset {match.name} for {name}")
        self._preset_cache[name] = (match.bank, match.preset)
        return (match.bank, match.preset)

    def _apply_preset(self, name: str) -> None:
        """Resolve instrument name and issue a MIDI program_select."""
        bp = self._resolve_preset(name)
        if bp is None or self._sfid is None or self._synth is None:
            return
        bank, preset = bp
        try:
            self._synth.program_select(CHAN, self._sfid, bank, preset)
            self._synth.cc(CHAN, 7, 127)   # CC 7 = channel volume: keep at max
            self._current_instrument = name
        except Exception as exc:
            print(f"[Audio] program_select failed for {name!r}: {exc}")

    def _play_or_update_locked(self, pitch: int, amplitude: float, instrument: str) -> None:
        """Must be called with ``self._synth_lock`` held."""
        assert self._synth is not None  # guaranteed by play_or_update's early return
        # Switch instrument if changed; end any current note first to avoid ghosts.
        if instrument != self._current_instrument:
            self._apply_preset(instrument)
            if self._current_pitch is not None:
                try:
                    self._synth.noteoff(CHAN, self._current_pitch)
                except Exception:
                    pass
                self._current_pitch = None

        # Start or retrigger note when pitch changes.
        if pitch != self._current_pitch:
            if self._current_pitch is not None:
                try:
                    self._synth.noteoff(CHAN, self._current_pitch)
                except Exception:
                    pass
            try:
                self._synth.noteon(CHAN, pitch, 127)  # velocity max; CC 11 handles volume
                self._current_pitch = pitch
            except Exception:
                pass

        # Smooth volume via MIDI expression (CC 11).
        vel = int(np.clip(amplitude, 0.0, 1.0) * 127)
        try:
            self._synth.cc(CHAN, 11, vel)
        except Exception:
            pass

    def _stop_note_locked(self) -> None:
        """Must be called with ``self._synth_lock`` held."""
        if self._current_pitch is not None and self._synth is not None:
            try:
                self._synth.noteoff(CHAN, self._current_pitch)
            except Exception:
                pass
            self._current_pitch = None
