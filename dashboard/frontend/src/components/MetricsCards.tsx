import { Clock3, Gauge, Route, Trophy } from "lucide-react";
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

  return (
    <article className={`metric-card latency-card ${className}${active ? " active-latency" : ""}`}>
      <div className="metric-card-top"><Clock3 size={20} /><span>{title}</span></div>
      <strong>{metric(summary?.latest_ms, " ms")}</strong>
      <small>{active ? "Current mode" : frameNumber ? `Last frame ${frameNumber}` : "Waiting for samples"}</small>
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

  return (
    <section className="metric-grid" aria-label="Current score metrics">
      <article className={`metric-card mode-card ${modeClass}`}>
        <div className="metric-card-top"><Gauge size={20} /><span>Processing mode</span></div>
        <strong>{mode}</strong>
        <small>Active inference location</small>
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
