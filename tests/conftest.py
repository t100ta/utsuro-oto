"""Pytest configuration: path setup and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so ``thereminvox`` is importable
# regardless of whether the package is installed in the active venv.
PROJECT_ROOT = Path(__file__).parents[1].resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def fresh_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Reset thereminvox.config to default state and redirect CONFIG_PATH to a temp dir.

    This prevents tests from writing to or reading from the real config.json
    inside the package, keeping tests hermetic and reproducible.
    """
    import thereminvox.config as cfg

    config_file = tmp_path / "config.json"
    monkeypatch.setattr(cfg, "CONFIG_PATH", config_file)
    monkeypatch.setattr(cfg, "_active_instruments", cfg.DEFAULT_ACTIVE_INSTRUMENTS.copy())
    monkeypatch.setattr(cfg, "_active_scale", "pentatonic_major")
    monkeypatch.setattr(cfg, "_active_instrument_idx", 0)

    return cfg
