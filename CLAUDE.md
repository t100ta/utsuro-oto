# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

UtsuroOto（虚空音）is a [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) app that turns the robot into a theremin: the camera detects hand position via MediaPipe, maps X→pitch and Y→amplitude, and renders audio through FluidSynth into the robot's built-in speaker via the SDK's GStreamer pipeline.

## Commands

```bash
# Install / setup
uv sync --dev               # install runtime + dev dependencies
# equivalently:
mise run install

# Development checks
uv run ruff check .         # lint
uv run ruff format .        # auto-format (line-length 119)
uv run mypy utsuro_oto      # type-check (mypy 1.18.2, Python 3.11)
uv run pytest               # run all tests
uv run pytest tests/test_sound_engine.py::TestSoundEngineHappyPath::test_first_play_or_update_starts_note  # single test
mise run check              # lint + typecheck + tests in one command

# Run app (requires Reachy Mini on network)
uv run python main.py
# or:
mise run app
```

Tests do **not** require hardware — all camera, FluidSynth, and SDK calls are mocked.

## Architecture

Three concurrent execution contexts in `utsuro_oto/main.py`:

| Context | What it does |
|---------|-------------|
| **Vision thread** | `reachy_mini.media.get_frame()` → `HandTracker` (MediaPipe) → smoothed `(x, y) ∈ [-1,1]` stored in `self._hand_pos` under `_lock`; annotates frames for MJPEG feed |
| **Audio/Motion thread** | 50 Hz loop: reads hand pos, applies `EMAFilter` → `HysteresisQuantizer` → pitch glide → `SoundEngine.set_voice()` + head tracking via `reachy_mini.set_target()` |
| **Main thread** | Attaches media pipeline, plays self-test tone, starts both daemon threads, serves FastAPI dashboard at `http://0.0.0.0:8042`, waits on `stop_event` |

### Module responsibilities

- **`utsuro_oto/main.py`** — `UtsuroOto(ReachyMiniApp)`: wires everything together; contains all FastAPI routes inline in `__init__`. Dashboard HTML is a string constant `_DASHBOARD_HTML`.
- **`utsuro_oto/sound_engine.py`** — `SoundEngine`: FluidSynth offline renderer (no hardware driver). Uses `scamp._dependencies.fluidsynth` (bundled binary) not the system one. PCM pump thread calls `get_samples()` at 16 kHz stereo and pushes `float32 (N, 2)` to `push_audio_sample()` — the only route to the physical speaker.
  - `set_voice(pitch_float, ...)` — legato path: holds `BASE_NOTE=66` (F#4) forever, uses MIDI pitch bend (±`PITCH_BEND_RANGE=24` st) for continuous glide. Never retriggers.
  - `play_or_update(pitch_int, ...)` — discrete path: issues `noteoff→noteon` on every pitch change.
- **`utsuro_oto/mapping.py`** — pure math, no I/O: `EMAFilter` (exponential moving average), `HysteresisQuantizer` (suppresses note-boundary jitter), `build_scale_notes`, `_aff` (affine interpolation), scale tables.
- **`utsuro_oto/hand_tracker.py`** — `HandTracker`: MediaPipe Hands wrapper. Flips the frame horizontally (`cv2.flip`) so movement feels mirror-like; normalizes landmarks to `[-1, 1]`. Uses `MIDDLE_FINGER_PIP` as palm center.
- **`utsuro_oto/config.py`** — thread-safe getters/setters for active scale and instrument; persisted to `utsuro_oto/config.json`. Loaded at module import time via `_load_initial_config()`. Tests must use the `fresh_config` fixture (see `tests/conftest.py`) to redirect `CONFIG_PATH` to a temp dir.
- **`utsuro_oto/fluidsynth_check.py`** — probes FluidSynth availability without crashing when `libfluidsynth` is absent.

### Key design decisions

- **Vision dropout**: on MediaPipe detection loss the hand position is held (not drifted to center). `IDLE_TIMEOUT = 1.5 s` after `_last_hand_seen` triggers genuine silence, not transient dropout.
- **Pitch bend range**: `PITCH_BEND_RANGE = 24` semitones set via RPN 0 after every `program_select` (some soundfonts reset it on preset change).
- **Audio pipeline**: `Synth.start()` is never called (no hardware driver). Only `get_samples()` + `push_audio_sample()` are used.
- **Config persistence**: `config.json` lives inside the package dir (`utsuro_oto/config.json`) and is updated on every dashboard change. It is also the file checked into the repo with defaults.

## Ruff configuration

Line length 119, quote style double, `select = ["E", "F", "W", "I", "C4", "D"]`. Docstring rules D100–D107, D203, D205, D213, D401 are ignored. Test files skip all `D` rules.
