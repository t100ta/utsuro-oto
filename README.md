---
title: UtsuroOto
emoji: 👋
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Play Reachy Mini like a theremin — hand position controls pitch and volume via camera tracking
tags:
 - reachy_mini
 - reachy_mini_python_app
---

<p align="center">
  <img src="assets/logo.png" alt="UtsuroOto（虚空音）" width="420">
</p>

# UtsuroOto（虚空音）

**Play [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) like a theremin using hand tracking.**

The robot's camera detects your hand in real time. Move it left/right to change pitch,
up/down to change volume. FluidSynth synthesizes audio and pushes it to the robot's
built-in speaker via the SDK media pipeline, while the robot head follows your hand.

---

## How to play

Stand in front of Reachy Mini and hold one hand up in front of the camera.

| Axis | Movement | Effect |
|------|----------|--------|
| **X** (horizontal) | Right → left | Low pitch → high pitch *(theremin convention — mirror of camera)* |
| **Y** (vertical)   | Top → bottom | Loud → silent |

- The note snaps to the active **scale** (no out-of-tune notes unless chromatic is selected).
- Hold still on a note to sustain it; move smoothly between notes to glide.
- Remove your hand for > 1.5 s → silence; robot head returns to neutral.

---

## Features

**Scales (6)**

| Key | Scale |
|-----|-------|
| `pentatonic_major` | 5-note major pentatonic *(default)* |
| `pentatonic_minor` | 5-note minor pentatonic |
| `natural_minor` | Natural / Aeolian minor |
| `natural_major` | Natural / Ionian major |
| `blues` | 6-note blues scale |
| `chromatic` | All 12 semitones (unquantized) |

**Instruments**

8 curated GM timbres are active by default (selectable on the dashboard):
`flute`, `choir_aahs`, `violin`, `synth_voice`, `warm_pad`, `trumpet`, `oboe`, `shakuhachi`.

All 128 General MIDI instruments are available. Up to 8 can be active at once
(`MAX_ACTIVE_INSTRUMENTS = 8`). Active selection and current scale are persisted to
`utsuro_oto/config.json` across restarts.

**MIDI range:** C3 (48) – C6 (84).

---

## Architecture

Three concurrent execution contexts:

```
Vision thread      camera frames → MediaPipe Hands → smoothed palm (X, Y) ∈ [-1, 1]
                   annotates frames → MJPEG feed at /video_feed

Audio/Motion thread  50 Hz loop:
                   EMA-smooth X/Y → X maps to MIDI pitch (right=high) →
                   HysteresisQuantizer snaps to scale notes →
                   Y maps to amplitude (top=loud) →
                   SoundEngine.play_or_update() → FluidSynth PCM →
                   push_audio_sample() → SDK GStreamer → speaker
                   + set_target(head=…) → robot head tracks hand

Main thread        attaches media pipeline, plays self-test tone,
                   starts both threads, serves FastAPI dashboard,
                   waits on stop_event, shuts down cleanly
```

**Audio pipeline detail:** FluidSynth runs in *offline* mode (no hardware driver).
A pump thread pulls `get_samples()` at 16 kHz stereo and pushes float32 PCM through
`reachy_mini.media.push_audio_sample()`, reusing the same GStreamer path as the
robot's built-in emotes. This is the only route that reaches the physical speaker.

---

## Install & run

**Prerequisites:** Python ≥ 3.10, [`uv`](https://docs.astral.sh/uv/),
[`mise`](https://mise.jdx.dev/), a Reachy Mini reachable on the network.

```bash
# Install runtime + dev dependencies
mise run install          # → uv sync --dev

# Launch (connects to Reachy Mini)
mise run app              # → uv run python main.py
# or directly:
python -m utsuro_oto.main
```

**On the robot:** installed as a `reachy_mini_apps` plugin and launched from the
Reachy Mini dashboard (entry point: `utsuro-oto = "utsuro_oto.main:UtsuroOto"`).

---

## Dashboard & API

The FastAPI dashboard is served by the SDK at **`http://0.0.0.0:8042`**.

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Dashboard HTML — live camera feed, note readout, scale/instrument selector |
| `GET` | `/status` | JSON: `{playing, pitch_midi, pitch_name, amplitude, hand_detected, instrument, scale, fluidsynth_ok}` |
| `GET` | `/video_feed` | MJPEG stream (`multipart/x-mixed-replace`) |
| `GET` | `/config` | JSON: `{scale, instrument, available_scales, available_instruments}` |
| `POST` | `/config` | Body: `{"scale": "…", "instrument_idx": N}` → returns updated config |

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UTSURO_OTO_FRAME_IS_RGB` | `0` | Set `1` if the camera backend already delivers RGB (skips BGR→RGB conversion) |
| `UTSURO_OTO_MODEL_COMPLEXITY` | `0` | MediaPipe model complexity: `0` (fast) or `1` (more accurate, higher CPU) |
| `UTSURO_OTO_MIN_DET_CONF` | `0.5` | MediaPipe min detection confidence (0.0–1.0). Lower detects more in poor lighting. |
| `UTSURO_OTO_SOUNDFONT` | `default` | Soundfont name or path. `default` uses Merlin.sf2 bundled with scamp. |
| `UTSURO_OTO_GAIN` | `0.5` | FluidSynth master gain (FluidSynth's own default 0.2 is quiet; raise to 1.0 if needed) |
| `UTSURO_OTO_AUDIO_TEST` | `1` | Set `0` to skip the 1.5 s startup self-test tone |

### Persisted runtime config (`utsuro_oto/config.json`)

Updated on every dashboard change; loaded at startup. Keys:
`active_instruments` (list), `active_scale` (string), `active_instrument_idx` (int).

### Tuning constants

Key control-loop constants are defined in `utsuro_oto/main.py` (lines 109–122):
`AUDIO_FREQ_HZ`, `IDLE_TIMEOUT`, `DEAD_ZONE`, head-tracking PID gains, EMA `alpha`, etc.

---

## Development

```bash
mise run test          # Run unit tests (no hardware required)
mise run lint          # Lint with ruff
mise run format        # Auto-format with ruff
mise run typecheck     # Type-check with mypy
mise run check         # lint + typecheck + tests (CI gate)
mise run app-check     # Validate app structure for the Reachy Mini dashboard
```

---

## Credits

- Instrument list and FluidSynth wiring adapted from
  [RemiFabre/Theremini](https://github.com/RemiFabre/Theremini).
- Hand tracking utilities adapted from
  [pollen-robotics/hand_tracker_v2](https://github.com/pollen-robotics/hand_tracker_v2).
