"use strict";

const phaseDescriptions = {
  cycle_start: "Clean conditions; cycle begins",
  cycle_end: "Clean conditions; cycle ends",
  bandwidth_200: "Bandwidth capped at 200 mbit",
  bandwidth_50: "Bandwidth capped at 50 mbit",
  bandwidth_5: "Bandwidth capped at 5 mbit",
  jitter_light: "Light delay and jitter applied",
  gpu_load: "GPU and Triton server under load",
  mixed: "5 mbit cap, packet loss, GPU load may be active",
  network_load: "Network delay, jitter, or packet loss active",
  combined: "Network and server load active",
  unknown: "Waiting for phase metrics",
};

const chartDefs = [
  { title: "Latency", unit: " ms", series: [{ label: "latency", color: "#147d92", source: "frames", key: "latency_ms" }] },
  { title: "Latency jitter", unit: " ms", series: [{ label: "jitter", color: "#d4560e", source: "frames", key: "jitter_ms" }] },
  { title: "Frame displacement", unit: " px", series: [{ label: "distance", color: "#d14a45", source: "frames", key: "displacement_px" }] },
  { title: "Cumulative displacement", unit: " px", series: [{ label: "score", color: "#b37916", source: "frames", key: "cumulative_displacement_px" }] },
  {
    title: "Placement mode",
    unit: "",
    min: 0,
    max: 1,
    series: [{ label: "local=0 fallback=0.5 remote=1", color: "#6559a8", source: "mode" }],
  },
  { title: "GPU utilization", unit: "%", min: 0, max: 100, series: [{ label: "GPU", color: "#26845a", source: "gpu", key: "gpu_util_pct" }] },
  {
    title: "Network conditions",
    unit: " ms",
    series: [
      { label: "delay", color: "#147d92", source: "network", key: "delay_ms" },
      { label: "jitter", color: "#b37916", source: "network", key: "jitter_ms" },
    ],
  },
];

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
    status.textContent = "Saving…";
    try {
      const result = await postJson("/api/save", { label });
      status.textContent = `Saved as ${result.stem}. Files: ${result.files.join(", ")}`;
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

async function loadInitialState() {
  try {
    const response = await fetch(apiUrl("/api/state"));
    if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
    render(await response.json());
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
      queueSocketRender(JSON.parse(event.data));
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

  el["force-local"].disabled = buttonsDisabled;
  el["force-remote"].disabled = buttonsDisabled;
  el["force-local"].classList.toggle("active", localActive);
  el["force-remote"].classList.toggle("active", remoteActive);

  let status = "Automatic SP-agent";
  if (manualEnabled) {
    if (pendingMode) status = "Sending command";
    else if (controlError) status = "Command failed";
    else if (control.requested_mode) status = `Requested ${String(control.requested_mode).toUpperCase()}`;
    else status = "Manual override";
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
  }
}

function renderCharts(state) {
  for (let index = 0; index < chartDefs.length; index += 1) {
    const def = chartDefs[index];
    const article = el["charts-grid"].children[index];
    article.querySelector("h3").textContent = def.title;
    const legend = article.querySelector(".chart-legend");
    legend.innerHTML = def.series.map((item) => `<span><i style="background:${item.color}"></i>${escapeHtml(item.label)}</span>`).join("");
    article.querySelector(".chart-body").innerHTML = sparkSvg(def, valuesForChart(def, state));
  }
}

function valuesForChart(def, state) {
  const history = state.history || {};
  return def.series.map((series) => {
    let items = [];
    if (series.source === "frames") items = history.frames || [];
    else if (series.source === "gpu") items = history.gpu || [];
    else if (series.source === "network") items = history.network || [];
    else if (series.source === "mode") {
      return {
        ...series,
        values: (history.frames || []).map((frame) => {
          if (String(frame.processing_mode || "").startsWith("remote")) return 1;
          if (frame.processing_mode === "local_fallback") return 0.5;
          if (frame.processing_mode === "local") return 0;
          return null;
        }),
      };
    }
    return {
      ...series,
      values: items.map((item) => typeof item[series.key] === "number" ? item[series.key] : null),
    };
  });
}

function sparkSvg(def, series) {
  const width = 420;
  const height = 150;
  const pad = 12;
  const values = series.flatMap((item) => item.values).filter((value) => value != null);
  if (!values.length) return `<div class="chart-empty">Waiting for metrics</div>`;

  const yMin = def.min ?? Math.min(...values);
  const yMax = def.max ?? Math.max(...values);
  const span = Math.max(yMax - yMin, 1);
  const longest = Math.max(...series.map((item) => item.values.length), 1);
  const point = (value, index) => {
    const x = pad + index / Math.max(longest - 1, 1) * (width - pad * 2);
    const y = height - pad - (value - yMin) / span * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };
  const lines = series.map((item) => {
    const points = item.values
      .map((value, index) => value == null ? null : point(value, index))
      .filter(Boolean)
      .join(" ");
    return `<polyline points="${points}" fill="none" stroke="${item.color}" stroke-width="3"></polyline>`;
  }).join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(def.title)} over time">
      <line x1="${pad}" x2="${width - pad}" y1="${height - pad}" y2="${height - pad}" class="chart-axis"></line>
      <line x1="${pad}" x2="${width - pad}" y1="${pad}" y2="${pad}" class="chart-gridline"></line>
      ${lines}
      <text x="${pad}" y="${pad + 11}">${yMax.toFixed(1)}${def.unit}</text>
      <text x="${pad}" y="${height - pad - 5}">${yMin.toFixed(1)}${def.unit}</text>
    </svg>
  `;
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
  const phaseOrder = ["cycle_start", "gpu_load", "jitter_light", "bandwidth_50", "mixed", "cycle_end"];
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
