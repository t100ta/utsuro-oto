"""Runtime configuration for ThereminVox.

Stores active instruments and active scale, persisted to ``config.json``
next to this file.  Thread-safe getters/setters are exposed so the dashboard
and the audio loop can read/write concurrently.

Adapted from RemiFabre/Theremini (instrument list) with added scale support.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List

from thereminvox.mapping import SCALE_TABLES

# ── All available GM instrument names (scamp-compatible) ───────────
ALL_INSTRUMENTS: List[str] = [
    "piano_1", "piano_2", "piano_3", "honky_tonk", "e_piano_1", "e_piano_2",
    "harpsichord", "clavinet", "celesta", "glockenspiel", "music_box",
    "vibraphone", "marimba", "xylophone", "tubular_bells", "dulcimer",
    "organ_1", "organ_2", "organ_3", "reed_organ", "accordion", "harmonica",
    "bandoneon", "nylon_guitar", "steel_guitar", "jazz_guitar", "clean_guitar",
    "muted_guitar", "overdrive_guitar", "distortion_guitar", "guitar_harmonics",
    "acoustic_bass", "fingered_bass", "picked_bass", "fretless_bass",
    "slap_bass_1", "slap_bass_2", "synth_bass_1", "synth_bass_2",
    "violin", "viola", "cello", "contrabass", "tremolo_strings",
    "pizzicato_strings", "harp", "timpani",
    "string_ensemble_1", "string_ensemble_2",
    "synth_strings_1", "synth_strings_2",
    "choir_aahs", "voice_oohs", "synth_voice", "orchestra_hit",
    "trumpet", "trombone", "tuba", "muted_trumpet", "french_horn",
    "brass_section", "synth_brass_1", "synth_brass_2",
    "soprano_sax", "alto_sax", "tenor_sax", "baritone_sax",
    "oboe", "english_horn", "bassoon", "clarinet",
    "piccolo", "flute", "recorder", "pan_flute", "blown_bottle",
    "shakuhachi", "whistle", "ocarina",
    "square_wave", "saw_wave", "synth_calliope", "chiffer_lead", "charang",
    "solo_voice", "fifth_saw", "bass_lead",
    "fantasia", "warm_pad", "polysynth", "space_voice", "bowed_glass",
    "metal_pad", "halo_pad", "sweep_pad",
    "ice_rain", "soundtrack", "crystal", "atmosphere", "brightness",
    "goblin", "echo_drops", "star_theme",
    "sitar", "banjo", "shamisen", "koto", "kalimba", "bagpipe",
    "fiddle", "shanai", "tinker_bell", "agogo", "steel_drum",
    "wood_block", "taiko_drum", "melodic_tom", "synth_drum",
    "reverse_cymbal", "guitar_fret_noise", "breath_noise",
    "seashore", "bird_tweet", "telephone_ring", "helicopter",
    "applause", "gunshot",
]

# Curated default set — good theremin-like timbres
DEFAULT_ACTIVE_INSTRUMENTS: List[str] = [
    "flute",
    "choir_aahs",
    "violin",
    "synth_voice",
    "warm_pad",
    "trumpet",
    "oboe",
    "shakuhachi",
]

MAX_ACTIVE_INSTRUMENTS = 8
ALL_SCALES = list(SCALE_TABLES.keys())

# ── Persistence ─────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_config_lock = threading.Lock()

# In-memory state (initialised by _load_initial_config below)
_active_instruments: List[str] = DEFAULT_ACTIVE_INSTRUMENTS.copy()
_active_scale: str = "pentatonic_major"
_active_instrument_idx: int = 0  # index into _active_instruments


def _persist() -> None:
    data = {
        "active_instruments": _active_instruments,
        "active_scale": _active_scale,
        "active_instrument_idx": _active_instrument_idx,
    }
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _clean_instrument_list(items: object) -> List[str]:
    if not isinstance(items, list):
        return DEFAULT_ACTIVE_INSTRUMENTS.copy()
    seen: set[str] = set()
    cleaned: List[str] = []
    for name in items:
        if not isinstance(name, str):
            continue
        name = name.strip()
        if name and name in ALL_INSTRUMENTS and name not in seen:
            seen.add(name)
            cleaned.append(name)
            if len(cleaned) >= MAX_ACTIVE_INSTRUMENTS:
                break
    return cleaned or DEFAULT_ACTIVE_INSTRUMENTS.copy()


def _load_initial_config() -> None:
    global _active_instruments, _active_scale, _active_instrument_idx
    if not CONFIG_PATH.exists():
        _persist()
        return
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _persist()
        return
    if isinstance(raw, dict):
        cleaned = _clean_instrument_list(raw.get("active_instruments"))
        if cleaned:
            _active_instruments = cleaned
        scale = raw.get("active_scale")
        if isinstance(scale, str) and scale in SCALE_TABLES:
            _active_scale = scale
        idx = raw.get("active_instrument_idx")
        if isinstance(idx, int) and 0 <= idx < len(_active_instruments):
            _active_instrument_idx = idx
    _persist()


# ── Public getters/setters ───────────────────────────────────────────

def get_active_instruments() -> List[str]:
    with _config_lock:
        return list(_active_instruments)


def set_active_instruments(new_items: List[str]) -> List[str]:
    cleaned = _clean_instrument_list(new_items)
    with _config_lock:
        global _active_instruments, _active_instrument_idx
        _active_instruments = cleaned
        _active_instrument_idx = min(_active_instrument_idx, len(cleaned) - 1)
        _persist()
        return list(_active_instruments)


def get_current_instrument() -> str:
    with _config_lock:
        return _active_instruments[_active_instrument_idx]


def set_instrument_idx(idx: int) -> str:
    with _config_lock:
        global _active_instrument_idx
        _active_instrument_idx = max(0, min(idx, len(_active_instruments) - 1))
        _persist()
        return _active_instruments[_active_instrument_idx]


def get_scale() -> str:
    with _config_lock:
        return _active_scale


def set_scale(name: str) -> str:
    with _config_lock:
        global _active_scale
        if name in SCALE_TABLES:
            _active_scale = name
            _persist()
        return _active_scale


def get_available_instruments() -> List[str]:
    return list(ALL_INSTRUMENTS)


_load_initial_config()

__all__ = [
    "get_active_instruments",
    "set_active_instruments",
    "get_current_instrument",
    "set_instrument_idx",
    "get_scale",
    "set_scale",
    "get_available_instruments",
    "ALL_INSTRUMENTS",
    "ALL_SCALES",
    "DEFAULT_ACTIVE_INSTRUMENTS",
    "MAX_ACTIVE_INSTRUMENTS",
]
