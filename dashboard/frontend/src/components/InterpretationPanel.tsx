import { Info } from "lucide-react";
import type { DashboardState } from "../types";

function interpretation(state: DashboardState): string {
  const gpu = state.infrastructure.gpu;
  const network = state.infrastructure.network;
  const frames = state.history.frames;
  const delay = Number(network.delay_ms || 0);
  const loss = Number(network.packet_loss_pct || 0);
  const utilization = Number(gpu.gpu_utilization_pct || 0);
  const queue = Number(gpu.triton_queue_duration_ms || 0);
  const recent = frames.slice(-8).map((frame) => frame.displacement_px).filter((value): value is number => value != null);

  if (delay >= 60 || loss >= 2) return "Network conditions are poor. Local processing may be safer until the path improves.";
  if (utilization >= 85 || queue >= 25) return "The GPU server is under load. Local processing may reduce queueing delay.";
  if (recent.length >= 6 && recent.slice(-3).reduce((sum, value) => sum + value, 0) > recent.slice(0, 3).reduce((sum, value) => sum + value, 0) * 1.35) {
    return "Displacement is rising. The prediction is lagging further behind the ground truth.";
  }
  if (state.latest.processing_mode === "remote" && utilization < 70 && delay < 30 && loss < 1) {
    return "Remote inference is currently beneficial: network and GPU conditions are both favorable.";
  }
  if (state.latest.processing_mode?.startsWith("local")) {
    return "Local inference avoids network and server variability, but watch whether its latency increases displacement.";
  }
  return "Watch latency and displacement together. A good placement decision keeps the prediction close to the ground truth.";
}

interface Props {
  state: DashboardState;
}

export function InterpretationPanel({ state }: Props) {
  return (
    <section className="panel interpretation-panel">
      <Info size={21} />
      <div>
        <h2>What does this mean?</h2>
        <p>{interpretation(state)}</p>
      </div>
    </section>
  );
}
