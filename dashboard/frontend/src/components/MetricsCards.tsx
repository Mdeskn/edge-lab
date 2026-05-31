import { Clock3, Gauge, Route, Trophy } from "lucide-react";
import type { DashboardState } from "../types";

const metric = (value?: number | null, suffix = "", digits = 1) =>
  value == null ? "N/A" : `${value.toFixed(digits)}${suffix}`;

interface Props {
  state: DashboardState;
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
      <article className="metric-card">
        <div className="metric-card-top"><Clock3 size={20} /><span>Frame latency</span></div>
        <strong>{metric(state.latest.latency_ms, " ms")}</strong>
        <small>Rolling avg {metric(state.latency.rolling_average_ms, " ms")}</small>
        <div className="metric-foot">
          <span>Min {metric(state.latency.min_ms, " ms")}</span>
          <span>P95 {metric(state.latency.p95_ms, " ms")}</span>
          <span>Max {metric(state.latency.max_ms, " ms")}</span>
        </div>
      </article>
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
