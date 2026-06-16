import { Info } from "lucide-react";
import type { DashboardState } from "../types";

function interpretation(state: DashboardState): string {
  const gpu = state.infrastructure.gpu;
  const network = state.infrastructure.network;
  const frames = state.history.frames;
  const delay = Number(network.delay_ms || 0);
  const loss = Number(network.packet_loss_pct || 0);
  const utilization = Number(gpu.gpu_util_pct || 0);
  const queue = Number(gpu.yolo_queue_ms || 0);
  const recent = frames.slice(-8).map((f) => f.displacement_px).filter((v): v is number => v != null);

  if (delay >= 60 || loss >= 2) {
    const parts: string[] = [];
    if (delay >= 60) parts.push(`delay ${delay.toFixed(0)} ms`);
    if (loss >= 2) parts.push(`packet loss ${loss.toFixed(1)}%`);
    return `Network path is degraded: ${parts.join(", ")}.`;
  }
  if (utilization >= 85 || queue >= 25) {
    return `GPU server is under load: ${utilization.toFixed(0)}% utilization, ${queue.toFixed(0)} ms queue time.`;
  }
  if (recent.length >= 6 && recent.slice(-3).reduce((a, v) => a + v, 0) > recent.slice(0, 3).reduce((a, v) => a + v, 0) * 1.35) {
    return "Displacement is rising: recent frames are tracking further from ground truth.";
  }
  if (state.latest.processing_mode === "remote") {
    return `Remote inference active. GPU: ${utilization.toFixed(0)}%, delay: ${delay.toFixed(0)} ms, loss: ${loss.toFixed(1)}%.`;
  }
  if (state.latest.processing_mode?.startsWith("local")) {
    return `Local inference active. GPU: ${utilization.toFixed(0)}%, delay: ${delay.toFixed(0)} ms.`;
  }
  return `GPU: ${utilization.toFixed(0)}%, delay: ${delay.toFixed(0)} ms, loss: ${loss.toFixed(1)}%.`;
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
