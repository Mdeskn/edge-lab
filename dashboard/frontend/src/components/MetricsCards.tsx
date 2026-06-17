import { Cloud, Clock3, Gauge, Laptop, Route, Trophy } from "lucide-react";
import { useState } from "react";
import { setPlacementMode } from "../api";
import type { DashboardState, LatencySummary } from "../types";

const metric = (value?: number | null, suffix = "", digits = 1) =>
  value == null ? "N/A" : `${value.toFixed(digits)}${suffix}`;

interface Props {
  state: DashboardState;
}

interface LatencyCardProps {
  title: string;
  summary?: LatencySummary;
  active: boolean;
  className: string;
}

function LatencyCard({ title, summary, active, className }: LatencyCardProps) {
  const frameNumber = summary?.frame_number;
  const label = summary?.source === "probe"
    ? frameNumber ? `Probe frame ${frameNumber}` : "Live probe"
    : active ? "Current mode" : frameNumber ? `Last frame ${frameNumber}` : "Waiting for samples";

  return (
    <article className={`metric-card latency-card ${className}${active ? " active-latency" : ""}`}>
      <div className="metric-card-top"><Clock3 size={20} /><span>{title}</span></div>
      <strong>{metric(summary?.latest_ms, " ms")}</strong>
      <small>{label}</small>
      <div className="metric-foot">
        <span>Avg {metric(summary?.rolling_average_ms, " ms")}</span>
        <span>P95 {metric(summary?.p95_ms, " ms")}</span>
        <span>N {summary?.sample_count ?? 0}</span>
      </div>
    </article>
  );
}

export function MetricsCards({ state }: Props) {
  const mode = (state.latest.processing_mode || "unknown").toUpperCase();
  const modeClass = mode.startsWith("LOCAL") ? "mode-local" : mode === "REMOTE" ? "mode-remote" : "";
  const manualControl = state.placement_control;
  const manualEnabled = Boolean(manualControl?.enabled);
  const [pendingMode, setPendingMode] = useState<"local" | "remote" | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);

  const requestPlacement = async (nextMode: "local" | "remote") => {
    setPendingMode(nextMode);
    setControlError(null);
    try {
      await setPlacementMode(nextMode);
    } catch (error) {
      setControlError(error instanceof Error ? error.message : "Placement command failed");
    } finally {
      setPendingMode(null);
    }
  };

  const placementStatus = !manualEnabled
    ? "Automatic SP-agent"
    : pendingMode
      ? "Sending command"
      : controlError
        ? "Command failed"
        : manualControl?.requested_mode
          ? `Requested ${manualControl.requested_mode.toUpperCase()}`
          : "Manual override";

  return (
    <section className="metric-grid" aria-label="Current score metrics">
      <article className={`metric-card mode-card ${modeClass}`}>
        <div className="metric-card-top"><Gauge size={20} /><span>Processing mode</span></div>
        <strong>{mode}</strong>
        <small>Active inference location</small>
        <div className="placement-controls" aria-label="Manual placement controls">
          <button
            type="button"
            title="Force local inference"
            className={`placement-button ${mode.startsWith("LOCAL") ? "active" : ""}`}
            disabled={!manualEnabled || pendingMode !== null}
            aria-pressed={mode.startsWith("LOCAL")}
            onClick={() => requestPlacement("local")}
          >
            <Laptop size={16} />
            <span>Local</span>
          </button>
          <button
            type="button"
            title="Force remote inference"
            className={`placement-button ${mode === "REMOTE" ? "active" : ""}`}
            disabled={!manualEnabled || pendingMode !== null}
            aria-pressed={mode === "REMOTE"}
            onClick={() => requestPlacement("remote")}
          >
            <Cloud size={16} />
            <span>Remote</span>
          </button>
        </div>
        <small className={`placement-status ${controlError ? "error" : ""}`}>{placementStatus}</small>
      </article>
      <LatencyCard
        title="Local latency"
        summary={state.latency.local}
        active={mode === "LOCAL"}
        className="local-latency"
      />
      <LatencyCard
        title="Remote latency"
        summary={state.latency.remote}
        active={mode === "REMOTE"}
        className="remote-latency"
      />
      <article className="metric-card score-card">
        <div className="metric-card-top"><Route size={20} /><span>Frame displacement</span></div>
        <strong>{metric(state.latest.displacement_px, " px")}</strong>
        <small>Rolling avg {metric(state.displacement.rolling_average_px, " px")}</small>
      </article>
      <article className="metric-card score-card">
        <div className="metric-card-top"><Trophy size={20} /><span>Cumulative score</span></div>
        <strong>{metric(state.latest.cumulative_displacement_px, " px", 0)}</strong>
        <small>Lower is better</small>
      </article>
    </section>
  );
}
