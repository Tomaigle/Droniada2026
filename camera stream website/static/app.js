/* app.js */

const POLL_MS   = 250;   // 4 Hz telemetry poll
const FPS_HIST  = 60;    // samples in fps chart

// ── FPS history ring buffer ────────────────────────────────────────────────

const fpsHistory = new Array(FPS_HIST).fill(0);

// ── Element refs ───────────────────────────────────────────────────────────

const dot      = document.getElementById("status-dot");
const fpsBadge = document.getElementById("fps-badge");

const sFps   = document.getElementById("s-fps");
const sDet   = document.getElementById("s-det");
const sModel = document.getElementById("s-model");
const sRes   = document.getElementById("s-res");
const sDepth = document.getElementById("s-depth");
const detList = document.getElementById("det-list");

const hdrModel = document.getElementById("hdr-model");
const hdrRes   = document.getElementById("hdr-res");
const hdrDepth = document.getElementById("hdr-depth");

// ── Colour → CSS data-attr map ─────────────────────────────────────────────

function guessColour(label) {
  const l = label.toLowerCase();
  if (l.includes("blue"))   return "blue";
  if (l.includes("red"))    return "red";
  if (l.includes("yellow")) return "yellow";
  if (l.includes("barrel")) return "barrel";
  return "unknown";
}

// ── Render detection list ──────────────────────────────────────────────────

function renderDetections(detections) {
  if (!detections || detections.length === 0) {
    detList.innerHTML = '<span class="none-msg">No detections</span>';
    return;
  }

  detList.innerHTML = detections.map(d => {
    const colour = guessColour(d.label);
    const depthStr = d.depth > 0 ? `${d.depth.toFixed(2)}m` : "—";
    return `
      <div class="det-item" data-colour="${colour}">
        <span class="det-label">${d.label}</span>
        <span class="det-conf">${(d.conf * 100).toFixed(0)}%</span>
        <span class="det-depth">${depthStr}</span>
      </div>`;
  }).join("");
}

// ── FPS sparkline chart ────────────────────────────────────────────────────

const canvas = document.getElementById("fps-chart");
const ctx    = canvas.getContext("2d");

function drawChart() {
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight || 80;

  ctx.clearRect(0, 0, W, H);

  const max  = Math.max(...fpsHistory, 1);
  const step = W / (FPS_HIST - 1);

  // Grid lines at 25%, 50%, 75%
  ctx.strokeStyle = "#1f2128";
  ctx.lineWidth   = 1;
  [0.25, 0.5, 0.75].forEach(f => {
    const y = H - f * H;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  });

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0,   "rgba(0,212,255,0.35)");
  grad.addColorStop(1,   "rgba(0,212,255,0)");

  ctx.beginPath();
  fpsHistory.forEach((v, i) => {
    const x = i * step;
    const y = H - (v / max) * H * 0.9;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  // Close fill path
  ctx.lineTo((FPS_HIST - 1) * step, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  fpsHistory.forEach((v, i) => {
    const x = i * step;
    const y = H - (v / max) * H * 0.9;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#00d4ff";
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  // Current value label
  const cur = fpsHistory[FPS_HIST - 1];
  ctx.fillStyle   = "#00d4ff";
  ctx.font        = "10px monospace";
  ctx.textAlign   = "right";
  ctx.fillText(`${cur.toFixed(1)} fps`, W - 4, 12);
}

// ── Poll ───────────────────────────────────────────────────────────────────

async function poll() {
  try {
    const res = await fetch("/api/telemetry");
    if (!res.ok) throw new Error("bad response");
    const d = await res.json();

    // Online
    dot.className = "dot dot--online";

    // Stats
    const fps = d.fps ?? 0;
    sFps.textContent   = fps.toFixed(1);
    sDet.textContent   = d.det_count ?? (d.detections?.length ?? 0);
    sModel.textContent = d.model   || "—";
    sRes.textContent   = d.resolution || "—";
    sDepth.textContent = d.depth_available ? "✓ active" : "✗ N/A";

    // Header meta
    hdrModel.textContent = d.model || "—";
    hdrRes.textContent   = d.resolution || "—";
    hdrDepth.textContent = d.depth_available ? "DEPTH ✓" : "DEPTH ✗";

    // FPS badge + history
    fpsBadge.textContent = `${fps.toFixed(1)} fps`;
    fpsHistory.shift();
    fpsHistory.push(fps);
    drawChart();

    // Detections
    renderDetections(d.detections);

  } catch (_) {
    dot.className     = "dot dot--error";
    fpsBadge.textContent = "offline";
  }
}

// ── Init ───────────────────────────────────────────────────────────────────

setInterval(poll, POLL_MS);
poll();

// Redraw chart on resize
window.addEventListener("resize", drawChart);
