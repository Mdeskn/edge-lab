import { Trophy } from "lucide-react";
import type { DashboardState } from "../types";

const show = (value?: number | null, suffix = "", digits = 1) =>
  value == null ? "N/A" : `${value.toFixed(digits)}${suffix}`;

interface Props {
  state: DashboardState;
}

export function SummaryPanel({ state }: Props) {
  const summary = state.summary;
  return (
    <section className="panel summary-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Current run</span>
          <h2>Run Summary</h2>
        </div>
        <Trophy size={20} />
      </div>
      <div className="summary-grid">
        <div><span>Total frames</span><strong>{summary.total_frames}</strong></div>
        <div><span>Local frames</span><strong>{summary.local_frames} <small>{show(summary.local_percentage, "%")}</small></strong></div>
        <div><span>Remote frames</span><strong>{summary.remote_frames} <small>{show(summary.remote_percentage, "%")}</small></strong></div>
        <div><span>Average latency</span><strong>{show(summary.average_latency_ms, " ms")}</strong></div>
        <div><span>Average displacement</span><strong>{show(summary.average_displacement_px, " px")}</strong></div>
        <div><span>Final score</span><strong>{show(summary.cumulative_displacement_px, " px", 0)}</strong></div>
        <div><span>Best / worst latency</span><strong>{show(summary.best_latency_ms, " ms")} / {show(summary.worst_latency_ms, " ms")}</strong></div>
        <div><span>Best / worst displacement</span><strong>{show(summary.best_displacement_px, " px")} / {show(summary.worst_displacement_px, " px")}</strong></div>
      </div>
    </section>
  );
}
