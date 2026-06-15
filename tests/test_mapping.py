"""Tests for thereminvox.mapping — core pitch/amplitude mapping logic.

These tests cover pure functions and classes that contain the musical heart
of ThereminVox.  No hardware, camera, or audio is needed.
"""

import pytest

from thereminvox.mapping import (
    MIDI_MAX,
    MIDI_MIN,
    SCALE_TABLES,
    EMAFilter,
    HysteresisQuantizer,
    _aff,
    build_scale_notes,
    midi_to_name,
)

# ── _aff (affine interpolation) ────────────────────────────────────

class TestAff:
    def test_midpoint(self):
        assert _aff(0.5, 0.0, 1.0, 0.0, 100.0) == pytest.approx(50.0)

    def test_clamp_low(self):
        assert _aff(-1.0, 0.0, 1.0, 10.0, 20.0) == 10.0

    def test_clamp_high(self):
        assert _aff(2.0, 0.0, 1.0, 10.0, 20.0) == 20.0

    def test_at_lower_bound(self):
        assert _aff(0.0, 0.0, 1.0, 5.0, 15.0) == 5.0

    def test_at_upper_bound(self):
        assert _aff(1.0, 0.0, 1.0, 5.0, 15.0) == 15.0

    def test_inverted_range(self):
        """y0 > y1: output should decrease as x increases."""
        result = _aff(0.25, 0.0, 1.0, 100.0, 0.0)
        assert result == pytest.approx(75.0)

    def test_midi_mapping(self):
        """Hand at center X (0) maps to midpoint of MIDI range."""
        mid = _aff(0.0, -1.0, 1.0, MIDI_MAX, MIDI_MIN)
        assert mid == pytest.approx((MIDI_MAX + MIDI_MIN) / 2)


# ── EMAFilter ────────────────────────────────────────────────────────

class TestEMAFilter:
    def test_first_sample_returned_unchanged(self):
        f = EMAFilter(alpha=0.3)
        assert f.update(42.0) == pytest.approx(42.0)

    def test_value_property_reflects_last_update(self):
        f = EMAFilter(alpha=0.5)
        f.update(10.0)
        f.update(20.0)
        assert f.value == pytest.approx(15.0)

    def test_convergence_toward_constant(self):
        f = EMAFilter(alpha=0.5)
        for _ in range(20):
            f.update(100.0)
        # After many iterations with constant input, value ≈ 100
        assert f.value == pytest.approx(100.0, abs=0.01)

    def test_reset_clears_state(self):
        f = EMAFilter(alpha=0.5)
        f.update(50.0)
        f.reset()
        assert f.value is None
        # After reset, first sample should again be returned unchanged
        assert f.update(7.0) == pytest.approx(7.0)

    def test_alpha_zero_holds_initial(self):
        """alpha=0 means no update — value stays at first sample."""
        f = EMAFilter(alpha=0.0)
        f.update(5.0)
        f.update(99.0)
        assert f.value == pytest.approx(5.0)

    def test_alpha_one_is_passthrough(self):
        """alpha=1 means always take new value."""
        f = EMAFilter(alpha=1.0)
        f.update(1.0)
        assert f.update(42.0) == pytest.approx(42.0)


# ── HysteresisQuantizer ───────────────────────────────────────────────

PENTA = [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84]


class TestHysteresisQuantizer:
    def test_initial_value_is_nearest(self):
        q = HysteresisQuantizer(hysteresis=0.4)
        assert q.quantize(49.0, PENTA) == 48  # 49 is closer to 48 than 50 (dist 1 vs 1)
        # 49 is equidistant — min() picks 48 (first in sorted list)

    def test_first_call_sets_current(self):
        q = HysteresisQuantizer(hysteresis=0.4)
        q.quantize(60.0, PENTA)
        assert q.current == 60

    def test_no_change_within_hysteresis(self):
        """Small oscillation near a note boundary must not flip the output."""
        q = HysteresisQuantizer(hysteresis=0.4)
        q.quantize(48.0, PENTA)  # anchor at 48
        # 48.3 is closer to 48 (dist 0.3) than to 50 (dist 1.7) — stays at 48
        assert q.quantize(48.3, PENTA) == 48

    def test_changes_when_clearly_past_boundary(self):
        """Sustained movement far past the midpoint must commit to the new note."""
        q = HysteresisQuantizer(hysteresis=0.4)
        q.quantize(48.0, PENTA)  # anchor at 48
        # 50.5: dist_from_48=2.5, dist_from_50=0.5.  2.5 > 0.5 + 0.4 → commit to 50
        assert q.quantize(50.5, PENTA) == 50

    def test_stays_at_current_when_barely_past_boundary(self):
        """Just at the midpoint: dist_current ≈ dist_closest, no commit."""
        q = HysteresisQuantizer(hysteresis=0.4)
        q.quantize(48.0, PENTA)
        # 49.0: dist_from_48=1.0, dist_from_50=1.0 → equidistant, won't commit
        result = q.quantize(49.0, PENTA)
        assert result == 48  # no commit because 1.0 is NOT > 1.0 + 0.4

    def test_empty_scale_falls_back_to_rounding(self):
        q = HysteresisQuantizer()
        assert q.quantize(48.6, []) == 49

    def test_single_note_scale_always_returns_that_note(self):
        q = HysteresisQuantizer()
        assert q.quantize(60.0, [72]) == 72
        assert q.quantize(0.0, [72]) == 72

    def test_reset_clears_current(self):
        q = HysteresisQuantizer(hysteresis=0.4)
        q.quantize(60.0, PENTA)
        q.reset()
        assert q.current is None
        # Next call should re-anchor at nearest note
        assert q.quantize(64.0, PENTA) == 64

    def test_current_property_reflects_committed_note(self):
        q = HysteresisQuantizer()
        q.quantize(72.0, PENTA)
        assert q.current == 72


# ── build_scale_notes ─────────────────────────────────────────────────

class TestBuildScaleNotes:
    def test_pentatonic_major_is_sorted(self):
        notes = build_scale_notes("pentatonic_major")
        assert notes == sorted(notes)

    def test_pentatonic_major_no_duplicates(self):
        notes = build_scale_notes("pentatonic_major")
        assert len(notes) == len(set(notes))

    def test_all_notes_within_range(self):
        for scale in SCALE_TABLES:
            notes = build_scale_notes(scale, MIDI_MIN, MIDI_MAX)
            for n in notes:
                assert MIDI_MIN <= n <= MIDI_MAX, f"{scale}: {n} out of range"

    def test_chromatic_has_all_semitones_in_range(self):
        notes = build_scale_notes("chromatic", 48, 60)
        assert set(notes) == set(range(48, 61))

    def test_pentatonic_minor_differs_from_major(self):
        major = set(build_scale_notes("pentatonic_major"))
        minor = set(build_scale_notes("pentatonic_minor"))
        assert major != minor

    def test_unknown_scale_falls_back_to_pentatonic_major(self):
        default = build_scale_notes("pentatonic_major")
        unknown = build_scale_notes("nonexistent_scale")
        assert unknown == default

    def test_custom_range(self):
        notes = build_scale_notes("pentatonic_major", midi_min=60, midi_max=72)
        assert all(60 <= n <= 72 for n in notes)

    def test_blues_scale_contains_blue_note(self):
        """The blues scale has a flat-5 (tritone) — the 'blue note'."""
        blues = build_scale_notes("blues", midi_min=48, midi_max=60)
        # C blues: C(48), Eb(51), F(53), F#(54), G(55), Bb(58)
        assert 54 in blues  # F# = C + tritone = blue note

    def test_natural_minor_has_seven_notes_per_octave(self):
        """Natural minor has 7 distinct pitch classes."""
        notes = build_scale_notes("natural_minor", 60, 71)
        assert len(notes) == 7


# ── midi_to_name ─────────────────────────────────────────────────────

class TestMidiToName:
    @pytest.mark.parametrize("midi,expected", [
        (60, "C4"),
        (48, "C3"),
        (84, "C6"),
        (61, "C♯4"),
        (69, "A4"),
        (57, "A3"),
    ])
    def test_known_notes(self, midi: int, expected: str):
        assert midi_to_name(midi) == expected
