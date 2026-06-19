/* ThereminVox dashboard — polls /status and /config, posts updates */

const STATUS_INTERVAL_MS = 200;  // 5 Hz status polling

let config = { scale: null, instrument: null, available_scales: [], available_instruments: [] };
let applyPending = false;

// ── Canvas XY indicator ──────────────────────────────────────────────
const canvas = document.getElementById("xy-canvas");
const ctx = canvas.getContext("2d");

function drawXY(x, y, handDetected) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  // Axis labels
  ctx.fillStyle = "#555";
  ctx.font = "11px monospace";
  ctx.fillText("← low pitch   high pitch →", 4, H - 4);
  ctx.save(); ctx.translate(8, H / 2 - 20); ctx.rotate(-Math.PI / 2);
  ctx.fillText("↑ loud", 0, 0); ctx.restore();

  if (!handDetected) return;

  // Map x/y from [-1,1] to canvas pixels
  const px = ((1.0 - (x + 1.0) / 2.0)) * W;   // mirror x so right = high pitch visually
  const py = ((y + 1.0) / 2.0) * H;

  // Glow ring
  const grad = ctx.createRadialGradient(px, py, 0, px, py, 20);
  grad.addColorStop(0, "rgba(100,200,255,0.6)");
  grad.addColorStop(1, "rgba(100,200,255,0)");
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.arc(px, py, 20, 0, 2 * Math.PI); ctx.fill();

  // Dot
  ctx.fillStyle = "#64c8ff";
  ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.fill();
}

// ── Status polling ───────────────────────────────────────────────────

async function pollStatus() {
  try {
    const resp = await fetch("/status");
    if (!resp.ok) return;
    const s = await resp.json();

    // Banner
    const banner = document.getElementById("status-banner");
    if (!s.fluidsynth_ok) {
      banner.textContent = "⚠ FluidSynth unavailable: " + (s.fluidsynth_error || "unknown error");
      banner.className = "banner error";
    } else if (s.playing) {
      banner.textContent = "🎵 Playing";
      banner.className = "banner ok";
    } else if (s.hand_detected) {
      banner.textContent = "👋 Hand detected — move up to increase volume";
      banner.className = "banner info";
    } else {
      banner.textContent = "Show your hand to the camera to play";
      banner.className = "banner idle";
    }

    // Readout values
    document.getElementById("note-name").textContent  = s.pitch_name || "--";
    document.getElementById("note-midi").textContent  = s.pitch_midi != null ? s.pitch_midi : "--";
    document.getElementById("amplitude").textContent  = s.amplitude != null
      ? (s.amplitude * 100).toFixed(0) + "%" : "--";
    document.getElementById("hand-x").textContent     = s.hand_x != null ? s.hand_x.toFixed(2) : "--";
    document.getElementById("hand-y").textContent     = s.hand_y != null ? s.hand_y.toFixed(2) : "--";

    // XY indicator
    drawXY(s.hand_x || 0, s.hand_y || 0, s.hand_detected);

  } catch (_) { /* backend starting or stopped */ }
}

// ── Config loading ───────────────────────────────────────────────────

async function loadConfig() {
  try {
    const resp = await fetch("/config");
    if (!resp.ok) return;
    config = await resp.json();

    populateSelect("scale-select", config.available_scales, config.scale);
    populateSelect("instr-select", config.available_instruments, config.instrument);
  } catch (_) {}
}

function populateSelect(id, options, selected) {
  const sel = document.getElementById(id);
  sel.replaceChildren();  // safe clear (avoids innerHTML)
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt.replace(/_/g, " ");
    if (opt === selected) el.selected = true;
    sel.appendChild(el);
  }
}

// ── Config apply ─────────────────────────────────────────────────────

document.getElementById("apply-btn").addEventListener("click", async () => {
  const scaleVal = document.getElementById("scale-select").value;
  const instrVal = document.getElementById("instr-select").value;
  const instrIdx = config.available_instruments.indexOf(instrVal);

  const statusEl = document.getElementById("config-status");
  statusEl.textContent = "Applying…";

  try {
    const resp = await fetch("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scale: scaleVal, instrument_idx: instrIdx >= 0 ? instrIdx : null }),
    });
    if (resp.ok) {
      const data = await resp.json();
      config.scale = data.scale;
      config.instrument = data.instrument;
      statusEl.textContent = `✓ Scale: ${data.scale}  Instrument: ${data.instrument}`;
    } else {
      statusEl.textContent = "Error applying config";
    }
  } catch (e) {
    statusEl.textContent = "Network error";
  }
});

// ── Boot ─────────────────────────────────────────────────────────────
loadConfig();
setInterval(pollStatus, STATUS_INTERVAL_MS);
pollStatus();
