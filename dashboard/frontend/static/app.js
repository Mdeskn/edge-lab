"use strict";

const phaseDescriptions = {
  cycle_start: "Clean conditions; cycle begins",
  cycle_end: "Clean conditions; cycle ends",
  bandwidth_20: "Bandwidth capped at 20 mbit",
  jitter_light: "Light delay and jitter applied",
  gpu_load: "GPU and Triton server under load",
  mixed: "20 mbit cap and GPU load active",
  network_load: "Network delay, jitter, or packet loss active",
  combined: "Network and server load active",
  unknown: "Waiting for phase metrics",
};

// Validated categorical palette (dataviz skill: fixed hue order, CVD/contrast
// checked). Colors are snapped to existing on-page conventions where one
// already exists (jitter = the .jitter-card orange, local/remote = the
// mode-card green/violet) so the charts agree with the rest of the dashboard.
const palette = {
  blue: "#2a78d6",
  aqua: "#1baf7a",
  amber: "#d9a735",
  green: "#26845a",
  violet: "#6559a8",
  red: "#d14a45",
  orange: "#d4560e",
};

const placementColors = {
  local: palette.green,
  local_fallback: palette.aqua,
  remote: palette.violet,
  unknown: "#a9b6b4",
};

const placementLabels = {
  local: "Local",
  local_fallback: "Fallback",
  remote: "Remote",
  unknown: "Unknown",
};

const phaseAbbrev = {
  cycle_start: "START",
  gpu_load: "GPU",
  jitter_light: "JITTER",
  bandwidth_20: "BW20",
  mixed: "MIXED",
  cycle_end: "END",
};

function abbreviatePhase(phase) {
  if (phaseAbbrev[phase]) return phaseAbbrev[phase];
  const bandwidthMatch = String(phase || "").match(/^bandwidth_(\d+)$/);
  if (bandwidthMatch) return `BW${bandwidthMatch[1]}`;
  return String(phase || "").slice(0, 6).toUpperCase();
}

const chartDefs = [
  { title: "Latency", unit: "ms", series: [{ label: "latency", color: palette.blue, source: "frames", key: "latency_ms" }] },
  { title: "Latency jitter", unit: "ms", series: [{ label: "jitter", color: palette.orange, source: "frames", key: "jitter_ms" }] },
  { title: "Frame displacement", unit: "px", series: [{ label: "distance", color: palette.red, source: "frames", key: "displacement_px" }] },
  { title: "Cumulative displacement", unit: "px", series: [{ label: "score", color: palette.amber, source: "frames", key: "cumulative_displacement_px" }] },
  { title: "Placement mode", kind: "state" },
  { title: "GPU utilization", unit: "%", min: 0, max: 100, series: [{ label: "GPU", color: palette.green, source: "gpu", key: "gpu_util_pct" }] },
  {
    title: "Network conditions",
    unit: "ms",
    series: [
      { label: "delay", color: palette.blue, source: "network", key: "delay_ms" },
      { label: "jitter", color: palette.orange, source: "network", key: "jitter_ms" },
    ],
  },
];

const CHART_W = 440;
const CHART_H = 170;
const PAD_L = 36;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 16;

const chartRuntime = [];
let tooltipEl = null;

const el = {};
let latestState = null;
let socket = null;
let stopped = false;
let pendingMode = null;
let controlError = "";
let videoStreamStarted = false;
let videoReconnectTimer = null;
let pendingSocketState = null;
let socketRenderTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  bindControls();
  buildCharts();
  loadInitialState();
  connectSocket();
});

function bindElements() {
  for (const node of document.querySelectorAll("[id]")) {
    el[node.id] = node;
  }
}

function bindControls() {
  el["force-local"].addEventListener("click", () => requestPlacement("local"));
  el["force-remote"].addEventListener("click", () => requestPlacement("remote"));
  el["force-auto"].addEventListener("click", () => requestPlacement("auto"));

  el["cycle-start-btn"].addEventListener("click", async () => {
    try { await postJson("/api/control/cycle", { action: "start_on_next_cycle" }); }
    catch (e) { alert("Failed to start: " + e.message); }
  });

  el["cycle-abort-btn"].addEventListener("click", async () => {
    if (!confirm("Abort the current cycle? All collected data will be discarded.")) return;
    try { await postJson("/api/control/cycle", { action: "abort_current_cycle" }); }
    catch (e) { alert("Failed to abort: " + e.message); }
  });

  el["cycle-refresh-duration-btn"].addEventListener("click", async () => {
    const button = el["cycle-refresh-duration-btn"];
    button.disabled = true;
    button.textContent = "Updating…";
    try {
      await postJson("/api/control/cycle", { action: "refresh_cycle_duration" });
      button.textContent = "Update requested";
      setTimeout(() => { button.textContent = "Update experiment time"; }, 1500);
    } catch (e) {
      button.textContent = "Update experiment time";
      alert("Failed to update experiment time: " + e.message);
    } finally {
      button.disabled = false;
    }
  });

  el["results-save-btn"].addEventListener("click", async () => {
    const label = el["results-label-input"].value;
    const status = el["results-save-status"];
    status.textContent = "Preparing charts…";
    let charts = {};
    let chartWarning = "";
    try {
      charts = await buildSavedCharts();
    } catch (e) {
      chartWarning = ` (charts not included: ${e.message})`;
    }
    status.textContent = "Saving…";
    try {
      const result = await postJson("/api/save", { label, charts });
      status.textContent = `Saved as ${result.stem}. Files: ${result.files.join(", ")}${chartWarning}`;
    } catch (e) {
      status.textContent = "Save failed: " + e.message;
    }
  });

  el["results-rerun-btn"].addEventListener("click", async () => {
    try {
      await postJson("/api/control/cycle", { action: "reset_to_armed" });
      el["results-modal"].hidden = true;
      el["results-label-input"].value = "";
      el["results-save-status"].textContent = "";
      _lastShownComplete = false;
    } catch (e) { alert("Failed to reset: " + e.message); }
  });
}

function apiUrl(path) {
  return path || "";
}

function wsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

// ---- Rolling chart history ----------------------------------------------
// The backend used to resend the entire rolling window (roughly 260 KB) on
// every WebSocket broadcast. It now sends a full window once, then only the
// records appended since the previous broadcast, and the browser keeps the
// window itself. Merging happens on receive rather than on render, because
// renders are coalesced and a dropped frame of rendering must not drop the
// history records that arrived with it.
const HISTORY_STREAMS = ["frames", "gpu", "network"];
const historyBuffers = { frames: [], gpu: [], network: [] };
let historySeq = 0;
let historyMax = 300;
let historySeeded = false;

// `authoritative` marks a snapshot that may replace what we already hold.
// Only the WebSocket is authoritative. The page also fetches /api/state at
// load so something renders before the socket opens, and those two race: if
// the socket seeds first and the slower fetch lands second, treating the fetch
// as authoritative would replace the buffer with older records AND rewind
// historySeq, so the next delta would start past the gap and those records
// would be lost for the rest of the session. That showed up as a hole in the
// charts near the start of a session.
function applyHistory(state, { authoritative = false } = {}) {
  if (typeof state.max_history === "number" && state.max_history > 0) {
    historyMax = state.max_history;
  }
  const incoming = state.history || {};

  if (state.history_mode === "delta" && historySeeded) {
    for (const stream of HISTORY_STREAMS) {
      const buffer = historyBuffers[stream];
      for (const record of incoming[stream] || []) {
        // A client that connected mid-interval sees an overlap between its
        // seeding snapshot and the first delta. Sequence numbers make the
        // duplicates cheap to drop.
        if ((record.__seq || 0) > historySeq) buffer.push(record);
      }
      const overflow = buffer.length - historyMax;
      if (overflow > 0) buffer.splice(0, overflow);
    }
  } else if (state.history_mode === "full" && (authoritative || !historySeeded)) {
    // The socket on connect or reconnect, or the initial fetch when the socket
    // has not seeded yet. A reconnect legitimately replaces everything, which
    // is also how the page recovers if the backend restarted and its sequence
    // began again from zero.
    for (const stream of HISTORY_STREAMS) {
      historyBuffers[stream] = (incoming[stream] || []).slice(-historyMax);
    }
    historySeeded = true;
  } else {
    if (state.history_mode === "delta") {
      // A delta with nothing to append it to. Rendering it alone would show a
      // few points that look overwritten rather than accumulated, so pull a
      // full window instead of displaying something misleading.
      resyncHistory();
    }
    // Otherwise a stale fetch landing after the socket seeded. Keep what we
    // have, and leave historySeq alone so the next delta still lines up.
    state.history = historyBuffers;
    return state;
  }

  if (typeof state.history_seq === "number") historySeq = state.history_seq;
  state.history = historyBuffers;
  return state;
}

let resyncPending = false;

async function resyncHistory() {
  if (resyncPending) return;
  resyncPending = true;
  try {
    const response = await fetch(apiUrl("/api/history"));
    if (!response.ok) return;
    const history = await response.json();
    applyHistory(
      {
        history: {
          frames: history.frames,
          gpu: history.gpu,
          network: history.network,
        },
        history_mode: "full",
        history_seq: history.history_seq,
        max_history: history.max_history,
      },
      { authoritative: true },
    );
    if (latestState) render(latestState);
  } catch (error) {
    // Best effort. The next socket reconnect sends a full window anyway.
  } finally {
    resyncPending = false;
  }
}

async function loadInitialState() {
  try {
    const response = await fetch(apiUrl("/api/state"));
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
    render(applyHistory(await response.json(), { authoritative: false }));
    hideError();
  } catch (error) {
    showError(`${error.message}. The dashboard will reconnect automatically.`);
  }
}

function connectSocket() {
  if (stopped) return;
  socket = new WebSocket(wsUrl());
  socket.onopen = () => setSocketConnected(true);
  socket.onmessage = (event) => {
    try {
      queueSocketRender(applyHistory(JSON.parse(event.data), { authoritative: true }));
      hideError();
    } catch (error) {
      console.error("Dashboard update failed", error);
      showError(`Dashboard update failed: ${error.message || "invalid state"}`);
      setSocketConnected(false);
    }
  };
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    setSocketConnected(false);
    if (!stopped) window.setTimeout(connectSocket, 1500);
  };
}

window.addEventListener("beforeunload", () => {
  stopped = true;
  if (socketRenderTimer) window.clearTimeout(socketRenderTimer);
  if (videoReconnectTimer) window.clearTimeout(videoReconnectTimer);
  if (socket) socket.close();
});

function queueSocketRender(state) {
  pendingSocketState = state;
  if (!latestState) {
    const initialState = pendingSocketState;
    pendingSocketState = null;
    render(initialState);
    return;
  }
  if (socketRenderTimer) return;

  socketRenderTimer = window.setTimeout(() => {
    socketRenderTimer = null;
    const newestState = pendingSocketState;
    pendingSocketState = null;
    if (newestState) render(newestState);
  }, 250);
}

async function requestPlacement(mode) {
  if (!latestState || !latestState.placement_control?.enabled || pendingMode) return;
  pendingMode = mode;
  controlError = "";
  renderPlacement(latestState);
  try {
    const response = await fetch(apiUrl("/api/placement"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || `Dashboard API returned ${response.status}`);
    }
  } catch (error) {
    controlError = error.message || "Placement command failed";
  } finally {
    pendingMode = null;
    renderPlacement(latestState);
  }
}

function render(state) {
  latestState = state;
  el["loading-state"].classList.add("hidden");
  el["dashboard-shell"].classList.remove("hidden");

  renderTopbar(state);
  renderVideo(state);
  renderMetrics(state);
  renderInfrastructure(state);
  renderInterpretation(state);
  renderCharts(state);
  renderSummary(state);
  renderCycleBanner(state);
  maybeShowResultsModal(state);
}

function renderTopbar(state) {
  const phase = state.experiment_phase || "unknown";
  text("phase-name", formatPhaseName(phase));
  text("phase-description", describePhase(phase));
}

function renderVideo(state) {
  text("frame-counter", `Frame ${state.frame?.frame_number ?? "N/A"}`);
  if (state.frame?.url && !videoStreamStarted) startVideoStream();

  const latest = state.latest || {};
  const display = state.frame || {};
  text("gt-coords", `GT (${coords(display.true_x ?? latest.true_x, display.true_y ?? latest.true_y)})`);
  const predictionFrame = display.prediction_frame_number;
  const predictionLabel = predictionFrame != null ? `Pred f${predictionFrame}` : "Pred";
  text(
    "pred-coords",
    `${predictionLabel} (${coords(display.predicted_x ?? latest.predicted_x, display.predicted_y ?? latest.predicted_y)})`,
  );
}

function startVideoStream() {
  if (stopped || videoStreamStarted) return;
  videoStreamStarted = true;

  const image = el["video-frame"];
  image.classList.add("is-visible");
  el["video-empty"].classList.add("hidden");
  image.onerror = () => {
    videoStreamStarted = false;
    if (!stopped && !videoReconnectTimer) {
      videoReconnectTimer = window.setTimeout(() => {
        videoReconnectTimer = null;
        startVideoStream();
      }, 1500);
    }
  };
  image.src = apiUrl(`/api/video-stream?v=${Date.now()}`);
}

function renderMetrics(state) {
  const mode = String(state.latest?.processing_mode || "unknown").toUpperCase();
  const modeCard = el["mode-card"];
  modeCard.classList.toggle("mode-local", mode.startsWith("LOCAL"));
  modeCard.classList.toggle("mode-remote", mode.startsWith("REMOTE"));
  text("processing-mode", mode);

  renderPlacement(state);
  renderLatency("local", state.latency?.local, mode === "LOCAL" || mode.startsWith("LOCAL"));
  renderLatency("remote", state.latency?.remote, mode.startsWith("REMOTE"));

  text("frame-displacement", metric(state.latest?.displacement_px, " px"));
  text("rolling-displacement", `Rolling avg ${metric(state.displacement?.rolling_average_px, " px")}`);
  text("cumulative-score", metric(state.latest?.cumulative_displacement_px, " px", 0));

  const jitter = state.latest?.jitter_ms;
  text("current-jitter", metric(jitter, " ms"));
  const jitterValues = (state.history?.frames || [])
    .map((frame) => frame.jitter_ms)
    .filter((value) => value != null && value > 0);
  const recentJitter = jitterValues.slice(-20);
  const avgJitter = recentJitter.length
    ? recentJitter.reduce((sum, value) => sum + value, 0) / recentJitter.length
    : null;
  text("jitter-avg", `Rolling avg ${metric(avgJitter, " ms")}`);
}

function renderPlacement(state) {
  const control = state.placement_control || {};
  const manualEnabled = Boolean(control.enabled);
  const mode = String(state.latest?.processing_mode || "unknown").toUpperCase();
  const localActive = mode.startsWith("LOCAL");
  const remoteActive = mode.startsWith("REMOTE");
  const buttonsDisabled = !manualEnabled || Boolean(pendingMode);
  // "auto" is a release command, not a placement, so it never lights up as
  // the active mode: it is highlighted while no manual lock is held.
  const requested = String(control.requested_mode || "").toLowerCase();
  const autoActive = requested === "" || requested === "auto";

  el["force-local"].disabled = buttonsDisabled;
  el["force-remote"].disabled = buttonsDisabled;
  el["force-auto"].disabled = buttonsDisabled || autoActive;
  el["force-local"].classList.toggle("active", localActive && !autoActive);
  el["force-remote"].classList.toggle("active", remoteActive && !autoActive);
  el["force-auto"].classList.toggle("active", autoActive);

  let status = "Automatic SP-agent";
  if (manualEnabled) {
    if (pendingMode) status = "Sending command";
    else if (controlError) status = "Command failed";
    else if (autoActive) status = "Automatic SP-agent";
    else status = `Locked to ${requested.toUpperCase()}`;
  }
  text("placement-status", status);
  el["placement-status"].classList.toggle("error", Boolean(controlError));
}

function renderLatency(mode, summary, active) {
  const prefix = `${mode}-latency`;
  const side = el[`${prefix}-card`];
  if (side) side.classList.toggle("is-active", active);
  text(prefix, metric(summary?.latest_ms, " ms"));
  text(`${prefix}-label`, latencyLabel(summary, active));
  text(`${prefix}-avg`, `Avg ${metric(summary?.rolling_average_ms, " ms")}`);
  text(`${prefix}-p95`, `P95 ${metric(summary?.p95_ms, " ms")}`);
  text(`${prefix}-n`, `N ${summary?.sample_count ?? 0}`);
}

function renderInfrastructure(state) {
  const gpu = state.infrastructure?.gpu || {};
  const network = state.infrastructure?.network || {};
  const kafka = state.infrastructure?.kafka || {};
  const dashboard = state.infrastructure?.dashboard || {};
  const memoryPct = number(gpu.gpu_mem_used_mb) != null && number(gpu.gpu_mem_total_mb)
    ? number(gpu.gpu_mem_used_mb) / number(gpu.gpu_mem_total_mb) * 100
    : null;

  text("gpu-util", show(gpu.gpu_util_pct, "%"));
  text("gpu-temp", show(gpu.gpu_temp_c, " °C"));
  text("gpu-memory", show(memoryPct, "%"));
  text("yolo-queue", show(gpu.yolo_queue_ms, " ms"));
  text("gpu-rps", show(gpu.total_rps, " /s"));

  text("net-mode", show(network.mode));
  text("net-bandwidth", show(network.bandwidth));
  text("net-delay", show(network.delay_ms, " ms"));
  text("net-jitter", show(network.jitter_ms, " ms"));
  text("net-loss", show(network.packet_loss_pct, "%"));

  setDot("kafka-dot", Boolean(kafka.connected));
  setDot("frames-dot", Boolean(dashboard.connected));
}

function renderInterpretation(state) {
  text("interpretation-text", interpretation(state));
}

function renderSummary(state) {
  const summary = state.summary || {};
  text("summary-total", summary.total_frames ?? 0);
  text("summary-local", `${summary.local_frames ?? 0} ${small(metric(summary.local_percentage, "%"))}`);
  text("summary-remote", `${summary.remote_frames ?? 0} ${small(metric(summary.remote_percentage, "%"))}`);
  text("summary-latency", metric(summary.average_latency_ms, " ms"));
  text("summary-displacement", metric(summary.average_displacement_px, " px"));
  text("summary-score", metric(summary.cumulative_displacement_px, " px", 0));
  text("summary-latency-range", `${metric(summary.best_latency_ms, " ms")} / ${metric(summary.worst_latency_ms, " ms")}`);
  text("summary-displacement-range", `${metric(summary.best_displacement_px, " px")} / ${metric(summary.worst_displacement_px, " px")}`);
}

function buildCharts() {
  const grid = el["charts-grid"];
  grid.textContent = "";
  for (let index = 0; index < chartDefs.length; index += 1) {
    const article = document.createElement("article");
    article.className = "chart-panel";
    article.id = `chart-${index}`;
    article.innerHTML = `
      <div class="chart-heading">
        <h3></h3>
        <div class="chart-legend"></div>
      </div>
      <div class="chart-body"></div>
    `;
    grid.appendChild(article);
    const body = article.querySelector(".chart-body");
    body.addEventListener("pointermove", (event) => onChartHover(event, index));
    body.addEventListener("pointerleave", hideTooltip);
  }
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.id = "chart-tooltip";
    tooltipEl.className = "chart-tooltip hidden";
    document.body.appendChild(tooltipEl);
  }
}

function renderCharts(state) {
  const history = state.history || {};
  for (let index = 0; index < chartDefs.length; index += 1) {
    const def = chartDefs[index];
    const article = el["charts-grid"].children[index];
    article.querySelector("h3").textContent = def.title;

    if (def.kind === "state") {
      renderLegend(article, [
        { label: placementLabels.local, color: placementColors.local },
        { label: placementLabels.local_fallback, color: placementColors.local_fallback },
        { label: placementLabels.remote, color: placementColors.remote },
      ]);
      const { svg, runtime } = stateStripSvg(def, history.frames || []);
      article.querySelector(".chart-body").innerHTML = svg;
      chartRuntime[index] = runtime;
      continue;
    }

    renderLegend(article, def.series);
    const series = valuesForChart(def, state);
    const items = sourceItemsFor(def, history);
    const showPhase = def.series.every((s) => s.source === "frames");
    const { svg, runtime } = lineChartSvg(def, series, items, showPhase);
    article.querySelector(".chart-body").innerHTML = svg;
    chartRuntime[index] = runtime;
  }
}

function renderLegend(article, items) {
  const legend = article.querySelector(".chart-legend");
  legend.innerHTML = items
    .map((item) => `<span><i style="background:${item.color}"></i>${escapeHtml(item.label)}</span>`)
    .join("");
}

function sourceItemsFor(def, history) {
  const source = def.series[0]?.source;
  if (source === "frames") return history.frames || [];
  if (source === "gpu") return history.gpu || [];
  if (source === "network") return history.network || [];
  return [];
}

function valuesForChart(def, state) {
  const items = sourceItemsFor(def, state.history || {});
  return def.series.map((series) => ({
    ...series,
    values: items.map((item) => (typeof item[series.key] === "number" ? item[series.key] : null)),
  }));
}

// ---- Shared layout helpers ----

function consecutiveRuns(items, keyFn) {
  if (!items.length) return [];
  const runs = [];
  let start = 0;
  let key = keyFn(items[0]);
  for (let i = 1; i <= items.length; i += 1) {
    const nextKey = i < items.length ? keyFn(items[i]) : null;
    if (i === items.length || nextKey !== key) {
      runs.push({ start, end: i - 1, key });
      start = i;
      key = nextKey;
    }
  }
  return runs;
}

function phaseBands(items) {
  return consecutiveRuns(items, (item) => item.experiment_phase || "unknown")
    .map((run) => ({ startIndex: run.start, endIndex: run.end, phase: run.key }));
}

function placementStateOf(item) {
  const mode = String(item?.processing_mode || "");
  if (mode.startsWith("remote")) return "remote";
  if (mode === "local_fallback") return "local_fallback";
  if (mode === "local") return "local";
  return "unknown";
}

function niceNumber(range, round) {
  if (!(range > 0)) return 1;
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / 10 ** exponent;
  let niceFraction;
  if (round) {
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;
  } else if (fraction <= 1) niceFraction = 1;
  else if (fraction <= 2) niceFraction = 2;
  else if (fraction <= 5) niceFraction = 5;
  else niceFraction = 10;
  return niceFraction * 10 ** exponent;
}

// Picks clean axis ticks and anchors magnitude metrics at zero, so a normal
// wobble in a tightly-scaled auto range never reads as a dramatic spike.
function niceAxis(dataMin, dataMax, tickCount) {
  let min = Number.isFinite(dataMin) ? dataMin : 0;
  let max = Number.isFinite(dataMax) ? dataMax : 1;
  if (min > 0) min = 0;
  if (min === max) max = min + 1;
  const step = niceNumber((max - min) / Math.max(tickCount - 1, 1), true);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.max(Math.ceil(max / step) * step, niceMin + step);
  const ticks = [];
  for (let v = niceMin; v <= niceMax + step * 0.5; v += step) ticks.push(Math.round(v * 1000) / 1000);
  return { min: niceMin, max: niceMax, ticks };
}

function formatTick(value) {
  if (value == null || Number.isNaN(value)) return "N/A";
  const abs = Math.abs(value);
  return abs !== 0 && abs < 10 ? value.toFixed(1) : String(Math.round(value));
}

function bandsSvg(bands, xAt, plotTop, plotH, plotRight) {
  return bands
    .map((band, i) => {
      const x1 = xAt(band.startIndex);
      const x2 = i < bands.length - 1 ? xAt(bands[i + 1].startIndex) : plotRight;
      const w = Math.max(x2 - x1, 0);
      const tint = i % 2 === 0 ? "chart-band-a" : "chart-band-b";
      const label = w >= 30
        ? `<text x="${(x1 + 4).toFixed(1)}" y="${(plotTop + 9).toFixed(1)}" class="chart-band-label">${escapeHtml(abbreviatePhase(band.phase))}</text>`
        : "";
      const boundary = i > 0
        ? `<line x1="${x1.toFixed(1)}" x2="${x1.toFixed(1)}" y1="${plotTop}" y2="${(plotTop + plotH).toFixed(1)}" class="chart-band-boundary"></line>`
        : "";
      return `<rect x="${x1.toFixed(1)}" y="${plotTop}" width="${w.toFixed(1)}" height="${plotH.toFixed(1)}" class="${tint}"></rect>${boundary}${label}`;
    })
    .join("");
}

function crosshairSvg(plotTop, plotBottom) {
  return `<line class="chart-crosshair" x1="0" y1="${plotTop}" x2="0" y2="${plotBottom.toFixed(1)}" visibility="hidden"></line>`;
}

// ---- Line / area charts (latency, jitter, displacement, GPU, network…) ----

function lineChartSvg(def, series, items, showPhase) {
  const allValues = series.flatMap((s) => s.values).filter((v) => v != null);
  if (!allValues.length) {
    return { svg: `<div class="chart-empty">Waiting for metrics</div>`, runtime: null };
  }

  const longest = Math.max(...series.map((s) => s.values.length), 1);
  const dataMin = def.min ?? Math.min(...allValues);
  const dataMax = def.max ?? Math.max(...allValues);
  const axis = niceAxis(dataMin, dataMax, 4);
  const span = Math.max(axis.max - axis.min, 1e-6);

  const plotW = CHART_W - PAD_L - PAD_R;
  const plotH = CHART_H - PAD_T - PAD_B;
  const plotRight = PAD_L + plotW;
  const plotBottom = PAD_T + plotH;
  const xAt = (index) => PAD_L + (longest <= 1 ? 0 : (index / (longest - 1)) * plotW);
  const yAt = (value) => PAD_T + plotH - ((value - axis.min) / span) * plotH;

  const bands = showPhase && items.length > 1 ? phaseBands(items) : [];

  const ticksSvg = axis.ticks
    .map((tick, i) => {
      const y = yAt(tick);
      const isTop = i === axis.ticks.length - 1;
      const gridline = tick > axis.min
        ? `<line x1="${PAD_L}" x2="${plotRight}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" class="chart-gridline"></line>`
        : "";
      const labelText = `${formatTick(tick)}${isTop && def.unit ? " " + def.unit : ""}`;
      return `${gridline}<text x="${(PAD_L - 6).toFixed(1)}" y="${(y + 3).toFixed(1)}" text-anchor="end" class="chart-tick-label">${escapeHtml(labelText)}</text>`;
    })
    .join("");

  const linesSvg = series
    .map((s) => {
      const points = s.values
        .map((v, i) => (v == null ? null : `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`))
        .filter(Boolean)
        .join(" ");
      let endMark = "";
      for (let i = s.values.length - 1; i >= 0; i -= 1) {
        if (s.values[i] == null) continue;
        const ex = xAt(i);
        const ey = yAt(s.values[i]);
        const labelText = `${formatTick(s.values[i])}${def.unit ? " " + def.unit : ""}`;
        const nearRight = ex > plotRight - 46;
        const lx = nearRight ? ex - 6 : ex + 8;
        endMark = `
          <circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="6" class="chart-end-ring"></circle>
          <circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="4" fill="${s.color}"></circle>
          <text x="${lx.toFixed(1)}" y="${(ey - 8).toFixed(1)}" text-anchor="${nearRight ? "end" : "start"}" class="chart-end-label">${escapeHtml(labelText)}</text>
        `;
        break;
      }
      return `<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></polyline>${endMark}`;
    })
    .join("");

  const svg = `
    <svg viewBox="0 0 ${CHART_W} ${CHART_H}" role="img" aria-label="${escapeHtml(def.title)} over time" preserveAspectRatio="none">
      ${bandsSvg(bands, xAt, PAD_T, plotH, plotRight)}
      ${ticksSvg}
      <line x1="${PAD_L}" x2="${plotRight}" y1="${plotBottom.toFixed(1)}" y2="${plotBottom.toFixed(1)}" class="chart-axis"></line>
      ${linesSvg}
      ${crosshairSvg(PAD_T, plotBottom)}
    </svg>
  `;

  const runtime = {
    kind: "line", def, series, items, showPhase, longest, xAt,
    plotLeft: PAD_L, plotRight, plotTop: PAD_T, plotBottom,
  };
  return { svg, runtime };
}

// ---- Placement mode: a categorical state strip, not a fake continuous line ----

function stateStripSvg(def, frames) {
  if (!frames.length) {
    return { svg: `<div class="chart-empty">Waiting for metrics</div>`, runtime: null };
  }

  const longest = frames.length;
  const plotW = CHART_W - PAD_L - PAD_R;
  const plotH = CHART_H - PAD_T - PAD_B;
  const plotRight = PAD_L + plotW;
  const plotBottom = PAD_T + plotH;
  const xAt = (index) => PAD_L + (longest <= 1 ? 0 : (index / (longest - 1)) * plotW);

  const bands = phaseBands(frames);
  const stripTop = PAD_T + plotH * 0.32;
  const stripH = plotH * 0.36;
  const segSvg = consecutiveRuns(frames, placementStateOf)
    .map((run) => {
      const x1 = xAt(run.start);
      const x2 = run.end + 1 < longest ? xAt(run.end + 1) : plotRight;
      const w = Math.max(x2 - x1 - 1, 1);
      const color = placementColors[run.key] || placementColors.unknown;
      return `<rect x="${x1.toFixed(1)}" y="${stripTop.toFixed(1)}" width="${w.toFixed(1)}" height="${stripH.toFixed(1)}" rx="2" fill="${color}"></rect>`;
    })
    .join("");

  const svg = `
    <svg viewBox="0 0 ${CHART_W} ${CHART_H}" role="img" aria-label="${escapeHtml(def.title)} over time" preserveAspectRatio="none">
      ${bandsSvg(bands, xAt, PAD_T, plotH, plotRight)}
      ${segSvg}
      ${crosshairSvg(PAD_T, plotBottom)}
    </svg>
  `;

  const runtime = {
    kind: "state", def, items: frames, showPhase: true, longest, xAt,
    plotLeft: PAD_L, plotRight, plotTop: PAD_T, plotBottom,
  };
  return { svg, runtime };
}

// ---- Hover: crosshair + shared tooltip ----

function onChartHover(event, index) {
  const runtime = chartRuntime[index];
  if (!runtime) return;
  const svg = event.currentTarget.querySelector("svg");
  if (!svg) return;
  const rect = svg.getBoundingClientRect();
  if (!rect.width) return;

  const userX = ((event.clientX - rect.left) / rect.width) * CHART_W;
  const clampedX = Math.min(Math.max(userX, runtime.plotLeft), runtime.plotRight);
  const frac = runtime.plotRight > runtime.plotLeft
    ? (clampedX - runtime.plotLeft) / (runtime.plotRight - runtime.plotLeft)
    : 0;
  const nearestIndex = Math.round(frac * Math.max(runtime.longest - 1, 0));

  const crosshair = svg.querySelector(".chart-crosshair");
  if (crosshair) {
    const x = runtime.xAt(nearestIndex).toFixed(1);
    crosshair.setAttribute("x1", x);
    crosshair.setAttribute("x2", x);
    crosshair.setAttribute("visibility", "visible");
  }

  showTooltip(event, runtime, nearestIndex);
}

function showTooltip(event, runtime, index) {
  if (!tooltipEl) return;
  const item = runtime.items[index];
  const frameLabel = item?.frame_number != null ? `Frame ${item.frame_number}` : `Sample ${index + 1}`;
  const phaseLabel = runtime.showPhase && item?.experiment_phase ? ` · ${formatPhaseName(item.experiment_phase)}` : "";

  let rows;
  if (runtime.kind === "state") {
    const state = placementStateOf(item);
    rows = `<div class="chart-tooltip-row"><span class="chart-tooltip-key"><i style="background:${placementColors[state]}"></i>${escapeHtml(placementLabels[state])}</span></div>`;
  } else {
    rows = runtime.series
      .map((s) => {
        const raw = s.values[index];
        const value = raw == null ? "N/A" : `${formatTick(raw)}${runtime.def.unit ? " " + runtime.def.unit : ""}`;
        return `<div class="chart-tooltip-row"><span class="chart-tooltip-key"><i style="background:${s.color}"></i>${escapeHtml(s.label)}</span><strong>${escapeHtml(value)}</strong></div>`;
      })
      .join("");
  }

  tooltipEl.innerHTML = `<div class="chart-tooltip-head">${escapeHtml(frameLabel)}${escapeHtml(phaseLabel)}</div>${rows}`;
  tooltipEl.classList.remove("hidden");

  const pad = 14;
  let left = event.clientX + pad;
  let top = event.clientY + pad;
  const tw = tooltipEl.offsetWidth;
  const th = tooltipEl.offsetHeight;
  if (left + tw > window.innerWidth - 8) left = event.clientX - tw - pad;
  if (top + th > window.innerHeight - 8) top = event.clientY - th - pad;
  tooltipEl.style.left = `${Math.max(8, left)}px`;
  tooltipEl.style.top = `${Math.max(8, top)}px`;
}

function hideTooltip() {
  if (tooltipEl) tooltipEl.classList.add("hidden");
  for (const article of el["charts-grid"]?.children || []) {
    article.querySelector(".chart-crosshair")?.setAttribute("visibility", "hidden");
  }
}

// ---- Full-cycle chart export (used by the Save results flow) ----
// The live charts only keep a rolling window (dashboard max_history), which
// is shorter than a full cycle at typical frame rates. /api/results reads
// the complete per-frame CSV instead, so saved charts cover the whole
// cycle_start -> cycle_end run. GPU/network charts are skipped: that data is
// only ever kept in the rolling in-memory buffer, never persisted, so an
// export of them would silently be a partial window, not the full cycle.
const SAVED_CHART_INDEX = { latency: 0, jitter: 1, displacement: 2, cumulative: 3, placement: 4 };

// The on-page charts take their axis, gridline, and label styling from
// styles.css. An exported file has no stylesheet, so those rules are inlined
// here. Without them the saved chart loses every stroke that a class supplies:
// the axis and gridlines come out invisible and the labels fall back to black.
const SAVED_CHART_STYLE = `
  text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .chart-axis { stroke: #c3c2b7; stroke-width: 1; }
  .chart-gridline { stroke: #e5eae9; stroke-width: 1; }
  .chart-band-a { fill: transparent; }
  .chart-band-b { fill: rgba(23, 35, 34, 0.035); }
  .chart-band-boundary { stroke: #e5eae9; stroke-width: 1; }
  .chart-band-label { fill: #8b9997; font-size: 8px; font-weight: 700; letter-spacing: 0.02em; }
  .chart-tick-label { fill: #708280; font-size: 9px; }
  .chart-end-ring { fill: #ffffff; }
  .chart-end-label { fill: #243433; font-size: 11px; font-weight: 700; }
  .chart-crosshair { display: none; }
`;

//: Exported at 2x so the chart stays sharp when scaled up in a report or slide.
const SAVED_CHART_SCALE = 2;

function standaloneChartSvg(svg) {
  // Replace the on-page root, which has no xmlns and inherits its size from
  // the layout. A rasteriser needs the namespace and explicit dimensions, and
  // an opaque background keeps the chart readable when pasted onto a slide.
  return svg.trim().replace(
    /^<svg[^>]*>/,
    `<svg xmlns="http://www.w3.org/2000/svg" width="${CHART_W}" height="${CHART_H}"`
      + ` viewBox="0 0 ${CHART_W} ${CHART_H}">`
      + `<style>${SAVED_CHART_STYLE}</style>`
      + `<rect width="${CHART_W}" height="${CHART_H}" fill="#ffffff"></rect>`,
  );
}

function chartSvgToPngBase64(svg) {
  return new Promise((resolve, reject) => {
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const image = new Image();

    image.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = CHART_W * SAVED_CHART_SCALE;
        canvas.height = CHART_H * SAVED_CHART_SCALE;
        const context = canvas.getContext("2d");
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/png").split(",")[1]);
      } catch (error) {
        reject(error);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("could not rasterise chart"));
    };
    image.src = url;
  });
}

async function buildSavedCharts() {
  const cs = latestState?.collection_state;
  const since = cs?.cycle_started_at;
  const until = cs?.cycle_completed_at;
  if (since == null || until == null) return {};

  const params = new URLSearchParams({ since: String(since), until: String(until) });
  const response = await fetch(apiUrl(`/api/results?${params.toString()}`));
  if (!response.ok) throw new Error(`could not load full-cycle results (${response.status})`);
  const { frames } = await response.json();
  if (!frames || !frames.length) return {};

  const wrappedState = { history: { frames } };
  const charts = {};
  for (const [key, index] of Object.entries(SAVED_CHART_INDEX)) {
    const def = chartDefs[index];
    const { svg } = def.kind === "state"
      ? stateStripSvg(def, frames)
      : lineChartSvg(def, valuesForChart(def, wrappedState), frames, true);
    if (!svg || !svg.trim().startsWith("<svg")) continue;
    charts[key] = await chartSvgToPngBase64(standaloneChartSvg(svg));
  }
  return charts;
}

function interpretation(state) {
  const gpu = state.infrastructure?.gpu || {};
  const network = state.infrastructure?.network || {};
  const frames = state.history?.frames || [];
  const delay = Number(network.delay_ms || 0);
  const loss = Number(network.packet_loss_pct || 0);
  const utilization = Number(gpu.gpu_util_pct || 0);
  const queue = Number(gpu.yolo_queue_ms || 0);
  const recent = frames.slice(-8).map((f) => f.displacement_px).filter((value) => value != null);

  if (delay >= 60 || loss >= 2) {
    const parts = [];
    if (delay >= 60) parts.push(`delay ${delay.toFixed(0)} ms`);
    if (loss >= 2) parts.push(`packet loss ${loss.toFixed(1)}%`);
    return `Network path is degraded: ${parts.join(", ")}.`;
  }
  if (utilization >= 85 || queue >= 25) {
    return `GPU server is under load: ${utilization.toFixed(0)}% utilization, ${queue.toFixed(0)} ms queue time.`;
  }
  if (
    recent.length >= 6
    && recent.slice(-3).reduce((sum, value) => sum + value, 0)
      > recent.slice(0, 3).reduce((sum, value) => sum + value, 0) * 1.35
  ) {
    return "Displacement is rising: recent frames are tracking further from ground truth.";
  }
  if (state.latest?.processing_mode?.startsWith("remote")) {
    return `Remote inference active. GPU: ${utilization.toFixed(0)}%, delay: ${delay.toFixed(0)} ms, loss: ${loss.toFixed(1)}%.`;
  }
  if (state.latest?.processing_mode?.startsWith("local")) {
    return `Local inference active. GPU: ${utilization.toFixed(0)}%, delay: ${delay.toFixed(0)} ms.`;
  }
  return `GPU: ${utilization.toFixed(0)}%, delay: ${delay.toFixed(0)} ms, loss: ${loss.toFixed(1)}%.`;
}

function setSocketConnected(connected) {
  setDot("socket-dot", connected);
  text("socket-label", connected ? "Live" : "Reconnecting");
}

function setDot(id, connected) {
  el[id]?.classList.toggle("is-connected", connected);
}

function showError(message) {
  el["error-banner"].textContent = message;
  el["error-banner"].classList.remove("hidden");
}

function hideError() {
  el["error-banner"].classList.add("hidden");
}

function formatPhaseName(phase) {
  if (!phase || phase === "unknown") return "UNKNOWN";
  return String(phase).replace(/_/g, " ").toUpperCase();
}

function describePhase(phase) {
  if (!phase || phase === "unknown") return phaseDescriptions.unknown;
  if (phaseDescriptions[phase]) return phaseDescriptions[phase];
  const bandwidthMatch = String(phase).match(/^bandwidth_(\d+)$/);
  if (bandwidthMatch) return `Bandwidth capped at ${bandwidthMatch[1]} mbit`;
  return "Custom experiment phase";
}

function latencyLabel(summary, active) {
  const frameNumber = summary?.frame_number;
  if (summary?.source === "probe") return frameNumber ? `Probe frame ${frameNumber}` : "Live probe";
  if (active) return "Current mode";
  return frameNumber ? `Last frame ${frameNumber}` : "Waiting for samples";
}

function metric(value, suffix = "", digits = 1) {
  return value == null || Number.isNaN(Number(value))
    ? "N/A"
    : `${Number(value).toFixed(digits)}${suffix}`;
}

function show(value, suffix = "") {
  if (value == null || value === "") return "N/A";
  return typeof value === "number" ? `${value.toFixed(1)}${suffix}` : `${value}${suffix}`;
}

function number(value) {
  return typeof value === "number" ? value : null;
}

function coords(x, y) {
  return x != null && y != null ? `${Number(x).toFixed(0)}, ${Number(y).toFixed(0)}` : "N/A";
}

function text(id, value) {
  if (el[id]) el[id].textContent = value;
}

function small(value) {
  return value === "N/A" ? "" : `(${value})`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`${resp.status}: ${txt}`);
  }
  return resp.json();
}

function renderCycleBanner(state) {
  const cs = state.collection_state || { state: "disconnected" };
  const phase = state.experiment_phase || "unknown";
  const banner = el["cycle-banner"];
  const startBtn = el["cycle-start-btn"];
  const abortBtn = el["cycle-abort-btn"];
  const progress = el["cycle-progress"];
  const progressText = el["cycle-progress-text"];
  const progressFill = el["cycle-progress-fill"];
  const total = (cs.cycle_duration_sec && cs.cycle_duration_sec > 0)
    ? cs.cycle_duration_sec
    : 75;

  banner.className = "cycle-banner cycle-banner-" + cs.state;
  text("cycle-state-label", cs.state.toUpperCase());

  startBtn.hidden = true;
  abortBtn.hidden = true;
  progress.hidden = true;
  progress.style.display = "none";

  if (cs.state === "disconnected") {
    text("cycle-state-detail", "Waiting for experiment infrastructure…");
  } else if (cs.state === "armed") {
    const duration = `${total} s`;
    progress.hidden = false;
    progress.style.display = "flex";
    if (progressText) progressText.textContent = `0 / ${total} s`;
    if (progressFill) progressFill.style.width = "0%";
    if (cs.armed_for_next_cycle) {
      text("cycle-state-detail", `Armed. Waiting for next cycle_start boundary. Current phase: ${phase}. Cycle duration: ${duration}.`);
    } else {
      text("cycle-state-detail", `Experiment running. Phase: ${phase}. Cycle duration: ${duration}. Click Start to collect on next cycle.`);
      startBtn.hidden = false;
    }
  } else if (cs.state === "collecting") {
    const startedAt = cs.cycle_started_at || 0;
    const elapsed = startedAt ? Math.max(0, (Date.now() / 1000) - startedAt) : 0;
    const pct = Math.min(100, (elapsed / total) * 100);
    progress.hidden = false;
    progress.style.display = "flex";
    abortBtn.hidden = false;
    if (progressText) progressText.textContent = `${Math.round(elapsed)} / ${total} s`;
    if (progressFill) progressFill.style.width = pct + "%";
    const seen = (cs.phases_seen || []).join(", ") || "none";
    text("cycle-state-detail", `Collecting cycle ${(cs.cycles_completed || 0) + 1}. Phase: ${phase}. Seen: ${seen}.`);
  } else if (cs.state === "complete") {
    const final = cs.final_cumulative_displacement;
    text("cycle-state-detail", `Cycle complete. Final score: ${final != null ? final.toFixed(2) + " px" : "N/A"}.`);
  }
}

let _lastShownComplete = false;

function maybeShowResultsModal(state) {
  const cs = state.collection_state || {};
  const modal = el["results-modal"];
  if (!modal) return;
  if (cs.state === "complete" && !_lastShownComplete) {
    _lastShownComplete = true;
    const score = cs.final_cumulative_displacement;
    text("results-score", score != null ? score.toFixed(2) + " px" : "N/A");
    renderResultsPhaseTable(state);
    modal.hidden = false;
  } else if (cs.state !== "complete") {
    _lastShownComplete = false;
    modal.hidden = true;
  }
}

function renderResultsPhaseTable(state) {
  const tbody = el["results-phase-tbody"];
  if (!tbody) return;
  const summary = state.phase_summary || {};
  const phaseOrder = state.phase_order || ["cycle_start", "gpu_load", "jitter_light", "bandwidth_20", "mixed", "cycle_end"];
  tbody.innerHTML = "";
  for (const phase of phaseOrder) {
    const s = summary[phase] || {};
    const frames = s.frames || 0;
    const meanLat = s.latency_frames > 0 ? (s.total_latency_ms / s.latency_frames).toFixed(1) : "—";
    const meanJit = s.jitter_frames > 0 ? (s.total_jitter_ms / s.jitter_frames).toFixed(1) : "—";
    const deadlineMisses = s.deadline_misses || 0;
    const disp = (s.total_displacement || 0).toFixed(1);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(phase)}</td>
      <td>${frames}</td>
      <td>${escapeHtml(meanLat)} ms</td>
      <td>${escapeHtml(meanJit)} ms</td>
      <td>${deadlineMisses}</td>
      <td>${escapeHtml(disp)} px</td>
    `;
    tbody.appendChild(tr);
  }
}
