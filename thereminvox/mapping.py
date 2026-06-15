"""Pitch/amplitude mapping utilities.

Provides:
- ``_aff``          — affine (linear) interpolation between two ranges.
- ``EMAFilter``     — exponential moving average for smoothing noisy input.
- ``HysteresisQuantizer`` — discrete pitch quantization with hysteresis to
                            suppress note-boundary jitter.
- ``build_scale_notes``   — expand a scale name into a list of MIDI note numbers.
- ``midi_to_name``        — pretty-print a MIDI note number.

Design note:
  Hand position comes in as [-1, 1] from MediaPipe (already normalized).
  X maps to pitch, Y maps to amplitude.  Both pass through an EMA before
  pitch is quantized via HysteresisQuantizer so that small trembling hands
  don't cause constant pitch changes at scale boundaries.
"""
from __future__ import annotations

from typing import Sequence

# ── Scale definitions (semitone offsets within one octave) ──────────
_PENTATONIC_MAJOR = [0, 2, 4, 7, 9]
_PENTATONIC_MINOR = [0, 3, 5, 7, 10]
_NATURAL_MINOR    = [0, 2, 3, 5, 7, 8, 10]
_NATURAL_MAJOR    = [0, 2, 4, 5, 7, 9, 11]
_CHROMATIC        = list(range(12))
_BLUES            = [0, 3, 5, 6, 7, 10]

SCALE_TABLES: dict[str, list[int]] = {
    "pentatonic_major": _PENTATONIC_MAJOR,
    "pentatonic_minor": _PENTATONIC_MINOR,
    "natural_minor":    _NATURAL_MINOR,
    "natural_major":    _NATURAL_MAJOR,
    "blues":            _BLUES,
    "chromatic":        _CHROMATIC,
}

MIDI_MIN = 48  # C3
MIDI_MAX = 84  # C6
NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]


def build_scale_notes(
    scale_name: str,
    midi_min: int = MIDI_MIN,
    midi_max: int = MIDI_MAX,
) -> list[int]:
    """Return sorted list of MIDI note numbers in *scale_name* within [midi_min, midi_max]."""
    offsets = SCALE_TABLES.get(scale_name, _PENTATONIC_MAJOR)
    notes: list[int] = []
    for root in range(0, 128, 12):
        for off in offsets:
            n = root + off
            if midi_min <= n <= midi_max:
                notes.append(n)
    return sorted(set(notes))


def midi_to_name(midi_note: int) -> str:
    """E.g. 60 → 'C4'."""
    return NOTE_NAMES[midi_note % 12] + str(midi_note // 12 - 1)


def _aff(x: float, a: float, b: float, y0: float, y1: float) -> float:
    """Affine (linear) interpolation: map x from [a,b] to [y0,y1], clamped."""
    if x <= a:
        return y0
    if x >= b:
        return y1
    return (x - a) * (y1 - y0) / (b - a) + y0


class EMAFilter:
    """Exponential Moving Average filter.

    Smooths noisy sensor input with a single ``alpha`` parameter:
    ``value = alpha * new_sample + (1 - alpha) * old_value``

    Lower alpha → more smoothing but more lag.
    Typical values: 0.10–0.25 for hand position.
    """

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self._value: float | None = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None

    @property
    def value(self) -> float | None:
        return self._value


class HysteresisQuantizer:
    """Quantize a continuous value to the nearest item in a discrete set with hysteresis.

    Without hysteresis, moving a hand near a scale-note boundary causes rapid
    alternation between two adjacent notes (jitter).  With hysteresis, the
    quantized output only changes when the continuous input has moved clearly
    past the boundary — i.e., when ``|input - current| > |input - next| + threshold``.

    Typical ``hysteresis`` value: 0.3–0.5 semitones.
    """

    def __init__(self, hysteresis: float = 0.4) -> None:
        self.hysteresis = hysteresis
        self._current: int | None = None

    def quantize(self, value: float, scale_notes: Sequence[int]) -> int:
        if not scale_notes:
            return int(round(value))

        closest = min(scale_notes, key=lambda n: abs(n - value))

        if self._current is None:
            self._current = closest
            return self._current

        if closest == self._current:
            return self._current

        # Commit to new note only when clearly past the boundary
        dist_from_current = abs(value - self._current)
        dist_from_closest = abs(value - closest)
        if dist_from_current > dist_from_closest + self.hysteresis:
            self._current = closest

        return self._current

    def reset(self) -> None:
        self._current = None

    @property
    def current(self) -> int | None:
        return self._current
