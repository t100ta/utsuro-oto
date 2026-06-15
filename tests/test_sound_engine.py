"""Tests for thereminvox.sound_engine — FluidSynth/scamp wrapper.

The tests cover two paths:
- FluidSynth unavailable (the expected state on any CI / dev machine without
  the system libfluidsynth installed) → engine must be gracefully silent.
- FluidSynth available (simulated with mocks) → note lifecycle is exercised.
"""

from unittest.mock import MagicMock, patch

from thereminvox.fluidsynth_check import FluidSynthProbeResult
from thereminvox.sound_engine import SoundEngine

# ── Helper: build a mocked SoundEngine with FluidSynth available ─────

def _make_engine_with_mocks(instrument: str = "flute"):
    """Return (engine, mock_session, mock_part) with scamp fully mocked."""
    probe_ok = FluidSynthProbeResult(True, None)
    mock_part = MagicMock(name="mock_part")
    mock_session = MagicMock(name="mock_session")
    mock_session.new_part.return_value = mock_part

    with (
        patch("thereminvox.sound_engine.probe_fluidsynth", return_value=probe_ok),
        patch("thereminvox.sound_engine.io"),  # silence stdout redirect
        patch("scamp.Session", return_value=mock_session),
    ):
        engine = SoundEngine(initial_instrument=instrument)

    # Patch the already-created session reference so future calls are tracked
    engine._session = mock_session
    engine._parts_cache = {instrument: mock_part}

    return engine, mock_session, mock_part


# ── FluidSynth unavailable (no-op path) ──────────────────────────────

class TestSoundEngineNoFluidSynth:
    def _engine(self) -> SoundEngine:
        probe_fail = FluidSynthProbeResult(False, "test: libfluidsynth not found")
        with patch("thereminvox.sound_engine.probe_fluidsynth", return_value=probe_fail):
            return SoundEngine()

    def test_ok_is_false(self):
        assert self._engine().ok is False

    def test_error_message_propagated(self):
        assert "libfluidsynth" in (self._engine().error or "")

    def test_play_or_update_is_silent(self):
        """Must not raise even when FluidSynth is absent."""
        engine = self._engine()
        engine.play_or_update(60, 0.8, "flute")  # should be a no-op

    def test_stop_note_is_silent(self):
        engine = self._engine()
        engine.stop_note()  # must not raise

    def test_shutdown_is_silent(self):
        engine = self._engine()
        engine.shutdown()  # must not raise


# ── FluidSynth available (happy path with mocks) ─────────────────────

class TestSoundEngineHappyPath:
    def test_ok_is_true(self):
        engine, _, _ = _make_engine_with_mocks()
        assert engine.ok is True

    def test_first_call_starts_note(self):
        engine, _, mock_part = _make_engine_with_mocks()
        mock_handle = MagicMock(name="note_handle")
        mock_part.start_note.return_value = mock_handle

        engine.play_or_update(60, 0.8, "flute")

        mock_part.start_note.assert_called_once_with(60, 0.8)
        assert engine._current_pitch == 60

    def test_same_pitch_updates_volume_only(self):
        engine, _, mock_part = _make_engine_with_mocks()
        mock_handle = MagicMock(name="note_handle")
        mock_part.start_note.return_value = mock_handle

        engine.play_or_update(60, 0.8, "flute")
        engine.play_or_update(60, 0.5, "flute")  # same pitch, different volume

        # start_note called only once (first call)
        mock_part.start_note.assert_called_once()
        # volume updated on second call
        mock_handle.change_note_volume.assert_called_with(0.5)
        # pitch NOT changed (stays same)
        mock_handle.change_note_pitch.assert_not_called()

    def test_pitch_change_calls_change_note_pitch(self):
        engine, _, mock_part = _make_engine_with_mocks()
        mock_handle = MagicMock(name="note_handle")
        mock_part.start_note.return_value = mock_handle

        engine.play_or_update(60, 0.8, "flute")
        engine.play_or_update(62, 0.8, "flute")  # pitch change

        mock_handle.change_note_pitch.assert_called_once_with(62)
        assert engine._current_pitch == 62

    def test_instrument_change_restarts_note(self):
        engine, _, mock_part = _make_engine_with_mocks("flute")  # noqa: F841
        violin_part = MagicMock(name="violin_part")
        engine._parts_cache["violin"] = violin_part

        flute_handle = MagicMock(name="flute_handle")
        violin_handle = MagicMock(name="violin_handle")
        mock_part.start_note.return_value = flute_handle
        violin_part.start_note.return_value = violin_handle

        engine.play_or_update(60, 0.8, "flute")
        engine.play_or_update(60, 0.8, "violin")  # instrument switch

        # Old note ended, new note started
        flute_handle.end.assert_called_once()
        violin_part.start_note.assert_called_once_with(60, 0.8)

    def test_stop_note_calls_end(self):
        engine, _, mock_part = _make_engine_with_mocks()
        mock_handle = MagicMock(name="note_handle")
        mock_part.start_note.return_value = mock_handle

        engine.play_or_update(60, 0.8, "flute")
        engine.stop_note()

        mock_handle.end.assert_called_once()
        assert engine._note_handle is None
        assert engine._current_pitch is None

    def test_double_stop_does_not_raise(self):
        engine, _, mock_part = _make_engine_with_mocks()
        engine.stop_note()  # no note playing → must not raise
        engine.stop_note()  # idempotent

    def test_shutdown_ends_note(self):
        engine, _, mock_part = _make_engine_with_mocks()
        mock_handle = MagicMock(name="note_handle")
        mock_part.start_note.return_value = mock_handle

        engine.play_or_update(60, 0.8, "flute")
        engine.shutdown()

        mock_handle.end.assert_called_once()
