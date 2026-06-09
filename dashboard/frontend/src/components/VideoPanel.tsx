import { useState, useEffect } from "react";
import { MonitorPlay } from "lucide-react";
import { resolveApiUrl } from "../api";
import type { DashboardState } from "../types";

interface Props {
  state: DashboardState;
}

export function VideoPanel({ state }: Props) {
  const nextUrl = resolveApiUrl(state.frame.url);
  const [displayedUrl, setDisplayedUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!nextUrl) return;
    const img = new Image();
    img.onload = () => setDisplayedUrl(nextUrl);
    img.src = nextUrl;
  }, [nextUrl]);

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
        {displayedUrl ? (
          <img src={displayedUrl} alt="Live annotated Edge-Lab frame" />
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
      <div className="video-coords">
        <span>GT ({state.latest.true_x != null && state.latest.true_y != null ? `${state.latest.true_x.toFixed(0)}, ${state.latest.true_y.toFixed(0)}` : "N/A"})</span>
        <span>Pred ({state.latest.predicted_x != null && state.latest.predicted_y != null ? `${state.latest.predicted_x.toFixed(0)}, ${state.latest.predicted_y.toFixed(0)}` : "N/A"})</span>
      </div>
    </section>
  );
}
