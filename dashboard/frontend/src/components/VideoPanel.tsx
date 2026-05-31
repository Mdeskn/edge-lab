import { MonitorPlay } from "lucide-react";
import { resolveApiUrl } from "../api";
import type { DashboardState } from "../types";

interface Props {
  state: DashboardState;
}

export function VideoPanel({ state }: Props) {
  const imageUrl = resolveApiUrl(state.frame.url);

  return (
    <section className="panel video-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Annotated stream</span>
          <h2>Live Video</h2>
        </div>
        <span className="frame-counter">Frame {state.frame.frame_number ?? "N/A"}</span>
      </div>
      <div className="video-stage">
        {imageUrl ? (
          <img key={state.frame.sequence} src={imageUrl} alt="Live annotated Edge-Lab frame" />
        ) : (
          <div className="video-empty">
            <MonitorPlay size={42} strokeWidth={1.5} />
            <span>Waiting for the first client frame</span>
          </div>
        )}
      </div>
      <div className="video-legend">
        <span><i className="legend-dot gt" />GT position</span>
        <span><i className="legend-dot pred" />Predicted position</span>
        <span><i className="legend-line" />Displacement</span>
      </div>
    </section>
  );
}
