"""Tests for utsuro_oto.sound_engine — offline FluidSynth renderer + SDK pump.

Architecture under test:
- FluidSynth renders PCM offline (get_samples, no audio driver opened).
- A pump thread pushes float32 (N, 2) PCM to reachy_mini.media.push_audio_sample().
- play_or_update / stop_note / shutdown control the MIDI channel via noteon/noteoff/cc.

Two test scenarios:
- FluidSynth unavailable (probe returns ok=False) → every method must be a no-op.
- FluidSynth available (mocked) → full note lifecycle, pump, attach_media.
"""

import threading
from unittest.mock import MagicMock, patch

import numpy as np

from utsuro_oto.fluidsynth_check import FluidSynthProbeResult
from utsuro_oto.sound_engine import CHAN, SoundEngine

# ── Test helpers ─────────────────────────────────────────────────────────────

def _make_mock_fluidsynth():
    """Return (mock_fs_module, mock_synth_instance)."""
    mock_synth = MagicMock(name="synth_instance")
    # get_samples returns int16 interleaved stereo: shape (2 * PUMP_CHUNK,)
    mock_synth.get_samples.return_value = np.zeros(2 * 1024, dtype=np.int16)

    mock_fs = MagicMock(name="fluidsynth_module")
    mock_fs.Synth.return_value = mock_synth
    return mock_fs, mock_synth


def _make_mock_preset(name: str = "Flute Gold", bank: int = 0, preset: int = 73) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.bank = bank
    m.preset = preset
    return m


def _make_engine_with_mocks(instrument: str = "flute"):
    """Return ``(engine, mock_synth)`` with scamp internals fully mocked.

    After this call the module-level patches are no longer active, but:
    - ``engine._synth`` still references ``mock_synth``.
    - ``engine._preset_cache`` is pre-populated so ``_resolve_preset`` never
      calls the real ``get_best_preset_match_for_name``.
    """
    probe_ok = FluidSynthProbeResult(True, None)
    mock_fs, mock_synth = _make_mock_fluidsynth()
    mock_preset = _make_mock_preset()

    with (
        patch("utsuro_oto.sound_engine.probe_fluidsynth", return_value=probe_ok),
        patch("utsuro_oto.sound_engine.fluidsynth", mock_fs),
        patch("utsuro_oto.sound_engine.resolve_soundfont", return_value="/fake/Merlin.sf2"),
        patch(
            "utsuro_oto.sound_engine.get_best_preset_match_for_name",
            return_value=(mock_preset, 0.9),
        ),
    ):
        engine = SoundEngine(initial_instrument=instrument)

    # Patches are gone but the synth object persists; keep preset cache populated
    # so later calls to _resolve_preset never hit the real scamp function.
    engine._preset_cache.update({
        "flute":      (0, 73),
        "violin":     (0, 40),
        "choir_aahs": (0, 52),
    })
    return engine, mock_synth


def _make_mock_media(rate: int = 16000) -> MagicMock:
    """Return a mock MediaManager with an `.audio` sub-object."""
    media = MagicMock(name="media")
    media.audio = MagicMock(name="media_audio")
    media.audio.get_output_audio_samplerate.return_value = rate
    return media


# ── FluidSynth unavailable (no-op path) ──────────────────────────────────────

class TestSoundEngineNoFluidSynth:
    def _engine(self) -> SoundEngine:
        probe_fail = FluidSynthProbeResult(False, "test: bundled fluidsynth not available")
        with patch("utsuro_oto.sound_engine.probe_fluidsynth", return_value=probe_fail):
            return SoundEngine()

    def test_ok_is_false(self):
        assert self._engine().ok is False

    def test_error_message_propagated(self):
        engine = self._engine()
        assert engine.error is not None
        assert "test" in engine.error

    def test_play_or_update_is_silent(self):
        self._engine().play_or_update(60, 0.8, "flute")  # must not raise

    def test_stop_note_is_silent(self):
        self._engine().stop_note()  # must not raise

    def test_shutdown_is_silent(self):
        self._engine().shutdown()  # must not raise

    def test_attach_media_is_silent(self):
        self._engine().attach_media(MagicMock())  # must not raise

    def test_self_test_no_op(self, monkeypatch):
        import utsuro_oto.sound_engine as se
        monkeypatch.setattr(se, "_AUDIO_TEST", True)
        self._engine().self_test()  # must not raise


# ── FluidSynth available — note lifecycle ─────────────────────────────────────

class TestSoundEngineHappyPath:
    def test_ok_is_true(self):
        engine, _ = _make_engine_with_mocks()
        assert engine.ok is True

    def test_first_play_or_update_starts_note(self):
        engine, mock_synth = _make_engine_with_mocks()
        engine.play_or_update(60, 0.8, "flute")
        mock_synth.noteon.assert_called_with(CHAN, 60, 127)
        assert engine._current_pitch == 60

    def test_same_pitch_updates_expression_only(self):
        engine, mock_synth = _make_engine_with_mocks()
        engine.play_or_update(60, 0.8, "flute")
        mock_synth.reset_mock()

        engine.play_or_update(60, 0.5, "flute")  # same pitch, different volume

        # noteon must NOT fire again
        mock_synth.noteon.assert_not_called()
        # CC 11 (expression) must be updated
        mock_synth.cc.assert_called_with(CHAN, 11, int(0.5 * 127))

    def test_pitch_change_triggers_noteoff_then_noteon(self):
        engine, mock_synth = _make_engine_with_mocks()
        engine.play_or_update(60, 0.8, "flute")
        mock_synth.reset_mock()

        engine.play_or_update(62, 0.8, "flute")

        mock_synth.noteoff.assert_called_with(CHAN, 60)
        mock_synth.noteon.assert_called_with(CHAN, 62, 127)
        assert engine._current_pitch == 62

    def test_instrument_change_reprogram_and_retrigger(self):
        engine, mock_synth = _make_engine_with_mocks("flute")
        engine.play_or_update(60, 0.8, "flute")
        mock_synth.reset_mock()

        engine.play_or_update(60, 0.8, "violin")

        # New instrument must be programmed
        mock_synth.program_select.assert_called()
        # Old note ended, new note started
        mock_synth.noteoff.assert_called()
        mock_synth.noteon.assert_called_with(CHAN, 60, 127)

    def test_stop_note_sends_noteoff_and_clears_pitch(self):
        engine, mock_synth = _make_engine_with_mocks()
        engine.play_or_update(60, 0.8, "flute")
        engine.stop_note()
        mock_synth.noteoff.assert_called_with(CHAN, 60)
        assert engine._current_pitch is None

    def test_double_stop_does_not_raise(self):
        engine, _ = _make_engine_with_mocks()
        engine.stop_note()  # no note playing
        engine.stop_note()  # idempotent

    def test_shutdown_ends_note_and_deletes_synth(self):
        engine, mock_synth = _make_engine_with_mocks()
        engine.play_or_update(60, 0.8, "flute")
        engine.shutdown()
        mock_synth.noteoff.assert_called()
        mock_synth.delete.assert_called_once()
        assert engine._synth is None

    # ── attach_media + pump ────────────────────────────────────────────────────

    def test_attach_media_calls_start_playing(self):
        engine, _ = _make_engine_with_mocks()
        media = _make_mock_media()

        engine.attach_media(media)
        engine.shutdown()

        media.start_playing.assert_called_once()

    def test_pump_pushes_float32_stereo_to_media(self):
        """Pump thread must push float32 (N, 2) arrays to media.push_audio_sample."""
        engine, _ = _make_engine_with_mocks()
        media = _make_mock_media()

        pushed = threading.Event()
        def _capture(data: object) -> None:
            pushed.set()
        media.push_audio_sample.side_effect = _capture

        engine.attach_media(media)
        assert pushed.wait(timeout=2.0), "pump never called push_audio_sample"
        engine.shutdown()

        call_arg = media.push_audio_sample.call_args[0][0]
        assert call_arg.dtype == np.float32
        assert call_arg.ndim == 2
        assert call_arg.shape[1] == 2

    def test_attach_media_with_no_audio_sub_object(self):
        """Gracefully disabled when media.audio is None."""
        engine, _ = _make_engine_with_mocks()
        media = MagicMock(name="media_no_audio")
        media.audio = None
        engine.attach_media(media)  # must not raise
        engine.shutdown()

    def test_shutdown_calls_stop_playing(self):
        engine, _ = _make_engine_with_mocks()
        media = _make_mock_media()
        engine.attach_media(media)
        engine.shutdown()
        media.stop_playing.assert_called_once()

    # ── self_test ──────────────────────────────────────────────────────────────

    def test_self_test_plays_and_ends_note(self, monkeypatch):
        """self_test must start a note, wait, then stop it."""
        import utsuro_oto.sound_engine as se
        monkeypatch.setattr(se, "_AUDIO_TEST", True)
        monkeypatch.setattr(se, "_SELF_TEST_DURATION", 0.0)  # skip sleep

        engine, mock_synth = _make_engine_with_mocks()
        media = _make_mock_media()
        engine.attach_media(media)
        engine.self_test()
        engine.shutdown()

        mock_synth.noteon.assert_called_with(CHAN, 60, 127)
        mock_synth.noteoff.assert_called_with(CHAN, 60)

    def test_self_test_skipped_when_audio_test_false(self, monkeypatch):
        """THEREMINVOX_AUDIO_TEST=0 must suppress the tone entirely."""
        import utsuro_oto.sound_engine as se
        monkeypatch.setattr(se, "_AUDIO_TEST", False)

        engine, mock_synth = _make_engine_with_mocks()
        media = _make_mock_media()
        engine.attach_media(media)
        engine.self_test()
        engine.shutdown()

        mock_synth.noteon.assert_not_called()

    def test_self_test_no_op_without_attach(self, monkeypatch):
        """self_test is skipped when attach_media has not been called yet."""
        import utsuro_oto.sound_engine as se
        monkeypatch.setattr(se, "_AUDIO_TEST", True)
        monkeypatch.setattr(se, "_SELF_TEST_DURATION", 0.0)

        engine, mock_synth = _make_engine_with_mocks()
        engine.self_test()  # no media attached → must be a no-op

        mock_synth.noteon.assert_not_called()
