"""ThereminVox — play Reachy Mini like a theremin using hand-tracking.

Architecture (2 threads + main thread):
  Vision thread  — camera frames → MediaPipe → smoothed hand position
  Audio/Motion thread — 50 Hz: hand pos → pitch/volume → FluidSynth + head tracking
  Main thread    — starts both, serves FastAPI dashboard, waits for stop_event

Hand coordinate system (from HandTracker):
  X ∈ [-1, 1]   right of camera = -1,  left = +1  (mirror-flipped)
  Y ∈ [-1, 1]   top of frame   = -1,  bottom = +1

Default mapping:
  X: -1 (right) → high pitch,  +1 (left) → low pitch  (theremin convention)
  Y: -1 (top)   → loud,        +1 (bottom) → silent
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp
from scipy.spatial.transform import Rotation as R

from thereminvox.config import (
    ALL_SCALES,
    get_active_instruments,
    get_current_instrument,
    get_scale,
    set_instrument_idx,
    set_scale,
)
from thereminvox.hand_tracker import HandTracker
from thereminvox.mapping import (
    EMAFilter,
    HysteresisQuantizer,
    _aff,
    build_scale_notes,
    midi_to_name,
)
from thereminvox.sound_engine import SoundEngine
from thereminvox.utils import allow_multiturn

# ── Dashboard HTML ──────────────────────────────────────────────────
_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>ThereminVox</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#1a1a2e;color:#eee;margin:0;padding:16px;max-width:640px}
h1{color:#7df;margin:0 0 12px;font-size:1.4em}
.card{background:#16213e;border-radius:8px;padding:12px;margin:8px 0}
.row{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
button{background:#0f3460;color:#ddd;border:1px solid #234;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:.85em;text-transform:capitalize}
button:hover{background:#e94560;border-color:#e94560}
button.on{background:#2a7a5a;border-color:#3ab;color:#fff}
.note{font-size:1.8em;font-weight:700;color:#7df;min-height:2em;line-height:1.2}
.sub{font-size:.8em;color:#89a;margin-bottom:4px}
img{width:100%;border-radius:8px;margin-top:8px}
</style>
</head>
<body>
<h1>ThereminVox</h1>
<div class="card">
  <div class="sub">Now playing</div>
  <div class="note" id="note">—</div>
  <div class="sub" id="info">—</div>
</div>
<div class="card">
  <div class="sub">Instrument</div>
  <div class="row" id="instrs"></div>
</div>
<div class="card">
  <div class="sub">Scale</div>
  <div class="row" id="scales"></div>
</div>
<img src="/video_feed" alt="camera feed">
<script>
var cfg={available_instruments:[],available_scales:[],instrument:"",scale:""};
function post(body){return fetch("/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(function(){return loadCfg();});}
function mkBtn(text,active,fn){var b=document.createElement("button");b.textContent=text.replace(/_/g," ");if(active)b.className="on";b.onclick=fn;return b;}
function render(){
  var ib=document.getElementById("instrs");ib.innerHTML="";
  cfg.available_instruments.forEach(function(n,i){ib.appendChild(mkBtn(n,n===cfg.instrument,function(){post({instrument_idx:i});}));});
  var sb=document.getElementById("scales");sb.innerHTML="";
  cfg.available_scales.forEach(function(n){sb.appendChild(mkBtn(n,n===cfg.scale,function(){post({scale:n});}));});
}
function loadCfg(){return fetch("/config").then(function(r){return r.json();}).then(function(c){cfg=c;render();});}
function poll(){
  fetch("/status").then(function(r){return r.json();}).then(function(s){
    document.getElementById("note").textContent=s.playing?(s.pitch_name+" (MIDI "+s.pitch_midi+")"):"—";
    document.getElementById("info").textContent=s.instrument+" · "+s.scale+" · amp "+((s.amplitude||0).toFixed(2))+" · "+(s.hand_detected?"hand ✓":"no hand");
    if(s.instrument!==cfg.instrument||s.scale!==cfg.scale){loadCfg();}
  }).catch(function(){});
}
loadCfg();
setInterval(poll,400);
</script>
</body>
</html>"""

# ── Control constants ────────────────────────────────────────────────
AUDIO_FREQ_HZ = 50          # control loop frequency
DEAD_ZONE = 0.05            # ignore hand motion smaller than this (fraction of [-1,1])
HEAD_PITCH_KP = 0.024       # head pitch gain (from hand_tracker_v2)
HEAD_YAW_KP = 0.028         # head yaw gain
HEAD_MAX_DELTA = 0.6        # max head rotation step per tick (radians)
ANT_MAX_DELTA = np.radians(5)  # max antenna step per tick
IDLE_TIMEOUT = 1.5          # seconds without hand before going silent/neutral
MIDI_MIN = 48               # C3
MIDI_MAX = 84               # C6
AMP_DEAD_ZONE = 0.03        # amplitude below this → silence (avoid noisy low notes)
# Frame resolution for vision loop (lower = faster on Wireless CM4)
VISION_WIDTH = 640
VISION_HEIGHT = 360


class Thereminvox(ReachyMiniApp):
    """Reachy Mini theremin app — hand X → pitch, hand Y → volume."""

    custom_app_url: str | None = "http://0.0.0.0:8042"
    # Use default media backend (camera + audio).  On Wireless, this will
    # use gstreamer for the camera stream; set to "gstreamer_no_video" if
    # you disable the camera feed to save CPU.
    request_media_backend: str | None = None

    def __init__(self) -> None:
        super().__init__()

        # ── Shared vision state (protected by _lock) ──────────────
        self._lock = threading.Lock()
        self._hand_pos: np.ndarray | None = None   # (x, y) in [-1, 1]
        self._last_hand_seen: float = 0.0
        self._last_frame: np.ndarray | None = None  # BGR frame for MJPEG

        # ── Audio smoothing / quantization ─────────────────────────
        self._ema_x = EMAFilter(alpha=0.15)
        self._ema_y = EMAFilter(alpha=0.15)
        self._quantizer = HysteresisQuantizer(hysteresis=0.4)

        # ── Sound engine ───────────────────────────────────────────
        initial_instr = get_current_instrument()
        self._sound = SoundEngine(initial_instrument=initial_instr)
        if not self._sound.ok:
            print(f"[ThereminVox] WARNING: FluidSynth unavailable — {self._sound.error}")
            print("[ThereminVox] Sound will be disabled; head tracking still works.")

        # ── Status payload (read by dashboard) ────────────────────
        self._status: dict[str, Any] = {
            "fluidsynth_ok": self._sound.ok,
            "fluidsynth_error": self._sound.error,
            "hand_detected": False,
            "hand_x": 0.0,
            "hand_y": 0.0,
            "pitch_midi": None,
            "pitch_name": "--",
            "amplitude": 0.0,
            "instrument": initial_instr,
            "scale": get_scale(),
            "playing": False,
        }

        # ── FastAPI endpoints ──────────────────────────────────────

        @self.settings_app.get("/status")
        def get_status() -> dict[str, Any]:
            return self._status

        @self.settings_app.get("/video_feed")
        def video_feed() -> StreamingResponse:
            return StreamingResponse(
                self._frame_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        class ConfigUpdate(BaseModel):
            scale: str | None = None
            instrument_idx: int | None = None

        @self.settings_app.post("/config")
        def update_config(cfg: ConfigUpdate) -> dict[str, Any]:
            if cfg.scale is not None:
                set_scale(cfg.scale)
                self._status["scale"] = get_scale()
            if cfg.instrument_idx is not None:
                instr = set_instrument_idx(cfg.instrument_idx)
                self._status["instrument"] = instr
            return {
                "scale": self._status["scale"],
                "instrument": self._status["instrument"],
                "available_scales": ALL_SCALES,
                "available_instruments": get_active_instruments(),
            }

        @self.settings_app.get("/config")
        def get_config() -> dict[str, Any]:
            return {
                "scale": get_scale(),
                "instrument": get_current_instrument(),
                "available_scales": ALL_SCALES,
                "available_instruments": get_active_instruments(),
            }

        @self.settings_app.get("/")
        def get_dashboard() -> HTMLResponse:
            return HTMLResponse(_DASHBOARD_HTML)

    # ── MJPEG helper ────────────────────────────────────────────────

    def _frame_generator(self):
        """Yield MJPEG frames from the latest camera capture."""
        while True:
            with self._lock:
                frame = self._last_frame
            if frame is None:
                time.sleep(0.05)
                continue
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(0.05)

    # ── Vision thread ────────────────────────────────────────────────

    def _vision_loop(
        self,
        reachy_mini: ReachyMini,
        stop_event: threading.Event,
    ) -> None:
        """Capture camera frames and run MediaPipe hand detection.

        model_complexity=0 is used for speed on the Wireless CM4.
        On a laptop, bumping to 1 gives better accuracy.
        """
        tracker = HandTracker(nb_hands=1, model_complexity=0)
        print("[Vision] Starting camera loop …")

        frame_count = 0
        hand_count = 0
        debug_t = time.time()

        while not stop_event.is_set():
            frame = reachy_mini.media.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (VISION_WIDTH, VISION_HEIGHT))
            hands = tracker.get_hands_positions(frame)

            frame_count += 1
            now = time.time()

            with self._lock:
                if hands:
                    hand_count += 1
                    palm = np.array(hands[0]["palm"], dtype=float)
                    self._hand_pos = palm
                    self._last_hand_seen = now
                else:
                    # Slowly drift hand_pos back toward center when lost
                    if self._hand_pos is not None:
                        self._hand_pos = self._hand_pos * 0.9
                self._last_frame = _annotate_frame(frame, hands)

            if now - debug_t >= 5.0:
                print(f"[Vision] frames={frame_count} hands={hand_count} in last 5s")
                frame_count = 0
                hand_count = 0
                debug_t = now

        print("[Vision] Stopped.")

    # ── Audio / Motion thread ────────────────────────────────────────

    def _audio_motion_loop(
        self,
        reachy_mini: ReachyMini,
        stop_event: threading.Event,
    ) -> None:
        """50 Hz control loop: hand pos → pitch/volume + head tracking."""
        euler_rot = np.array([0.0, 0.0, 0.0])   # roll, pitch, yaw
        head_pose = np.eye(4)
        prev_antennas = np.array([0.0, 0.0])
        antennas: list[float] = [0.0, 0.0]      # always list[float] (allow_multiturn return type)
        is_idle = True

        print("[Audio/Motion] Starting 50 Hz control loop …")

        while not stop_event.is_set():
            t0 = time.time()

            with self._lock:
                hand_pos = self._hand_pos.copy() if self._hand_pos is not None else None
                time_since_hand = t0 - self._last_hand_seen

            idle = hand_pos is None or time_since_hand > IDLE_TIMEOUT

            if idle:
                # ── Idle: silence + slowly return head to neutral ──
                if not is_idle:
                    self._sound.stop_note()
                    self._ema_x.reset()
                    self._ema_y.reset()
                    self._quantizer.reset()
                    is_idle = True
                    self._status["playing"] = False
                    print("[Audio/Motion] Idle — hand lost.")

                # Gentle return to neutral (5% per tick)
                euler_rot += np.clip(-euler_rot, -0.05, 0.05)
                antennas = allow_multiturn([0.0, 0.0], list(prev_antennas), ANT_MAX_DELTA)

            else:
                # ── Active: theremin mapping ───────────────────────
                is_idle = False
                assert hand_pos is not None  # guaranteed: idle=False means hand_pos is not None

                # EMA smoothing of raw hand position
                sx = self._ema_x.update(hand_pos[0])
                sy = self._ema_y.update(hand_pos[1])

                # X → MIDI pitch (continuous float before quantization)
                # X=-1 (right side of camera) → high pitch (theremin convention)
                midi_float = _aff(sx, -1.0, 1.0, MIDI_MAX, MIDI_MIN)

                # Quantize to active scale with hysteresis
                scale_notes = build_scale_notes(get_scale(), MIDI_MIN, MIDI_MAX)
                pitch = self._quantizer.quantize(midi_float, scale_notes)

                # Y → amplitude: top (-1) = loud, bottom (+1) = silent
                amplitude = _aff(sy, -1.0, 1.0, 1.0, 0.0)
                amplitude = float(np.clip(amplitude, 0.0, 1.0))

                # Play or update sound
                if amplitude > AMP_DEAD_ZONE and self._sound.ok:
                    instrument = get_current_instrument()
                    self._sound.play_or_update(pitch, amplitude, instrument)
                    self._status["playing"] = True
                else:
                    self._sound.stop_note()
                    self._status["playing"] = False

                # Update status for dashboard
                self._status.update({
                    "hand_detected": True,
                    "hand_x": round(float(sx), 3),
                    "hand_y": round(float(sy), 3),
                    "pitch_midi": pitch,
                    "pitch_name": midi_to_name(pitch),
                    "amplitude": round(amplitude, 3),
                    "instrument": get_current_instrument(),
                    "scale": get_scale(),
                })

                # Head tracking — follow hand at low gain
                error = np.array([0.0, 0.0]) - hand_pos   # error = center - pos
                error[np.abs(error) < DEAD_ZONE] = 0.0
                error = np.clip(error, -HEAD_MAX_DELTA, HEAD_MAX_DELTA)

                euler_rot += np.array([
                    0.0,
                    -HEAD_PITCH_KP * error[1],   # pitch: Y error
                    HEAD_YAW_KP  * error[0],   # yaw:   X error
                ])
                euler_rot = np.clip(
                    euler_rot,
                    [0.0, -np.deg2rad(30), -np.deg2rad(170)],
                    [0.0,  np.deg2rad(20),  np.deg2rad(170)],
                )

                antennas = allow_multiturn([0.0, 0.0], list(prev_antennas), ANT_MAX_DELTA)

            # Apply head pose — catch WebSocket errors on shutdown
            try:
                head_pose[:3, :3] = R.from_euler("xyz", euler_rot).as_matrix()
                reachy_mini.set_target(head=head_pose, antennas=np.array(antennas))
                prev_antennas = np.array(antennas)
            except Exception:
                break

            elapsed = time.time() - t0
            time.sleep(max(0.0, 1.0 / AUDIO_FREQ_HZ - elapsed))

        print("[Audio/Motion] Stopped.")

    # ── Main entry point ─────────────────────────────────────────────

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        print("=" * 60)
        print("  ThereminVox — hand tracking theremin")
        print(f"  FluidSynth: {'OK' if self._sound.ok else 'UNAVAILABLE'}")
        print(f"  Scale:      {get_scale()}")
        print(f"  Instrument: {get_current_instrument()}")
        print(f"  Dashboard:  {self.custom_app_url}")
        print("=" * 60)

        # Wire up the SDK media pipeline and start the PCM pump thread.
        # attach_media must come before self_test so the pump is running when
        # the test tone is played.
        self._sound.attach_media(reachy_mini.media)

        # Play a short tone to confirm audio routing before the main loop starts.
        self._sound.self_test()

        reachy_mini.enable_motors()
        try:
            t_vision = threading.Thread(
                target=self._vision_loop,
                args=(reachy_mini, stop_event),
                daemon=True,
            )
            t_audio = threading.Thread(
                target=self._audio_motion_loop,
                args=(reachy_mini, stop_event),
                daemon=True,
            )
            t_vision.start()
            t_audio.start()

            # Wait for stop signal
            stop_event.wait()

            t_vision.join(timeout=3.0)
            t_audio.join(timeout=3.0)
        finally:
            # shutdown() stops the pump thread and calls media.stop_playing().
            self._sound.shutdown()
            reachy_mini.disable_motors()
            print("[ThereminVox] Stopped cleanly.")


# ── Annotation helper (not in separate file to keep deps minimal) ────

def _annotate_frame(
    frame: np.ndarray,
    hands: list[dict] | None,
) -> np.ndarray:
    """Draw palm and index tip markers on the frame for the dashboard preview."""
    if not hands:
        return frame
    h, w = frame.shape[:2]
    for hand in hands:
        for key, color in (("palm", (0, 0, 255)), ("index_tip", (0, 255, 0))):
            pos = hand.get(key)
            if pos is None:
                continue
            # Inverse of _norm: map [-1,1] back to pixel coords
            px = int((1.0 - (pos[0] + 1.0) / 2.0) * w)
            py = int((pos[1] + 1.0) / 2.0 * h)
            cv2.circle(frame, (px, py), 6, color, -1)
    return frame


if __name__ == "__main__":
    app = Thereminvox()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
