"""Tests for thereminvox.config — instrument/scale persistence and validation."""

import json

from thereminvox.config import DEFAULT_ACTIVE_INSTRUMENTS

# ── Scale getters/setters ─────────────────────────────────────────────

class TestScale:
    def test_default_scale_is_pentatonic_major(self, fresh_config):
        assert fresh_config.get_scale() == "pentatonic_major"

    def test_set_valid_scale(self, fresh_config):
        result = fresh_config.set_scale("natural_minor")
        assert result == "natural_minor"
        assert fresh_config.get_scale() == "natural_minor"

    def test_set_invalid_scale_is_ignored(self, fresh_config):
        fresh_config.set_scale("natural_minor")
        result = fresh_config.set_scale("___nonexistent___")
        assert result == "natural_minor"  # unchanged
        assert fresh_config.get_scale() == "natural_minor"

    def test_set_scale_persists_to_json(self, fresh_config):
        fresh_config.set_scale("blues")
        data = json.loads(fresh_config.CONFIG_PATH.read_text())
        assert data["active_scale"] == "blues"


# ── Instrument index getters/setters ──────────────────────────────────

class TestInstrumentIdx:
    def test_default_instrument(self, fresh_config):
        assert fresh_config.get_current_instrument() == DEFAULT_ACTIVE_INSTRUMENTS[0]

    def test_set_valid_idx(self, fresh_config):
        instr = fresh_config.set_instrument_idx(1)
        assert instr == DEFAULT_ACTIVE_INSTRUMENTS[1]
        assert fresh_config.get_current_instrument() == DEFAULT_ACTIVE_INSTRUMENTS[1]

    def test_idx_clamped_low(self, fresh_config):
        instr = fresh_config.set_instrument_idx(-99)
        assert instr == DEFAULT_ACTIVE_INSTRUMENTS[0]

    def test_idx_clamped_high(self, fresh_config):
        n = len(DEFAULT_ACTIVE_INSTRUMENTS)
        instr = fresh_config.set_instrument_idx(n + 100)
        assert instr == DEFAULT_ACTIVE_INSTRUMENTS[n - 1]

    def test_set_idx_persists_to_json(self, fresh_config):
        fresh_config.set_instrument_idx(2)
        data = json.loads(fresh_config.CONFIG_PATH.read_text())
        assert data["active_instrument_idx"] == 2


# ── Active instrument list ────────────────────────────────────────────

class TestActiveInstruments:
    def test_default_active_instruments(self, fresh_config):
        assert fresh_config.get_active_instruments() == DEFAULT_ACTIVE_INSTRUMENTS

    def test_set_valid_instruments(self, fresh_config):
        result = fresh_config.set_active_instruments(["flute", "violin"])
        assert result == ["flute", "violin"]
        assert fresh_config.get_active_instruments() == ["flute", "violin"]

    def test_invalid_instrument_names_are_removed(self, fresh_config):
        result = fresh_config.set_active_instruments(["flute", "not_real_instrument", "violin"])
        assert result == ["flute", "violin"]

    def test_duplicates_are_removed(self, fresh_config):
        result = fresh_config.set_active_instruments(["flute", "flute", "violin"])
        assert result == ["flute", "violin"]

    def test_max_eight_instruments(self, fresh_config):
        many = ["flute", "violin", "trumpet", "oboe", "clarinet", "cello", "harp", "piano_1", "piano_2"]
        result = fresh_config.set_active_instruments(many)
        assert len(result) <= 8

    def test_empty_list_falls_back_to_defaults(self, fresh_config):
        result = fresh_config.set_active_instruments([])
        assert result == DEFAULT_ACTIVE_INSTRUMENTS

    def test_all_invalid_falls_back_to_defaults(self, fresh_config):
        result = fresh_config.set_active_instruments(["fake1", "fake2"])
        assert result == DEFAULT_ACTIVE_INSTRUMENTS

    def test_persists_to_json(self, fresh_config):
        fresh_config.set_active_instruments(["flute", "violin"])
        data = json.loads(fresh_config.CONFIG_PATH.read_text())
        assert data["active_instruments"] == ["flute", "violin"]


# ── Persistence round-trip ────────────────────────────────────────────

class TestPersistenceRoundTrip:
    def test_settings_survive_reload(self, fresh_config):
        """After writing config.json, _load_initial_config should restore state."""
        fresh_config.set_scale("natural_major")
        fresh_config.set_active_instruments(["oboe", "trumpet"])
        fresh_config.set_instrument_idx(1)

        # Simulate a reload by resetting to defaults then calling _load_initial_config
        fresh_config._active_scale = "pentatonic_major"
        fresh_config._active_instruments = DEFAULT_ACTIVE_INSTRUMENTS.copy()
        fresh_config._active_instrument_idx = 0

        fresh_config._load_initial_config()

        assert fresh_config.get_scale() == "natural_major"
        assert fresh_config.get_active_instruments() == ["oboe", "trumpet"]
        assert fresh_config.get_current_instrument() == "trumpet"
