import { Cpu, Network, Radio, Server } from "lucide-react";
import type { DashboardState, MetricValue } from "../types";
import { StatusDot } from "./StatusDot";

const show = (value: MetricValue, suffix = "") =>
  value == null || value === "" ? "N/A" : `${typeof value === "number" ? value.toFixed(1) : value}${suffix}`;

interface Props {
  state: DashboardState;
}

export function InfrastructurePanel({ state }: Props) {
  const { gpu, network, kafka, dashboard } = state.infrastructure;
  const memoryPct =
    typeof gpu.gpu_mem_used_mb === "number" && typeof gpu.gpu_mem_total_mb === "number" && gpu.gpu_mem_total_mb
      ? (gpu.gpu_mem_used_mb as number) / (gpu.gpu_mem_total_mb as number) * 100
      : null;

  return (
    <section className="panel infrastructure-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Live signals</span>
          <h2>Infrastructure</h2>
        </div>
        <Radio size={20} />
      </div>
      <div className="infra-columns">
        <div className="infra-group">
          <h3><Cpu size={17} /> GPU server</h3>
          <dl>
            <div><dt>GPU utilization</dt><dd>{show(gpu.gpu_util_pct, "%")}</dd></div>
            <div><dt>GPU temp</dt><dd>{show(gpu.gpu_temp_c, " °C")}</dd></div>
            <div><dt>GPU memory</dt><dd>{show(memoryPct, "%")}</dd></div>
            <div><dt>YOLO queue</dt><dd>{show(gpu.yolo_queue_ms, " ms")}</dd></div>
            <div><dt>Requests/s</dt><dd>{show(gpu.total_rps, " /s")}</dd></div>
          </dl>
        </div>
        <div className="infra-group">
          <h3><Network size={17} /> Network path</h3>
          <dl>
            <div><dt>Mode</dt><dd>{show(network.mode)}</dd></div>
            <div><dt>Bandwidth</dt><dd>{show(network.bandwidth)}</dd></div>
            <div><dt>Delay</dt><dd>{show(network.delay_ms, " ms")}</dd></div>
            <div><dt>Jitter</dt><dd>{show(network.jitter_ms, " ms")}</dd></div>
            <div><dt>Packet loss</dt><dd>{show(network.packet_loss_pct, "%")}</dd></div>
          </dl>
        </div>
        <div className="infra-group status-group">
          <h3><Server size={17} /> Data links</h3>
          <StatusDot connected={kafka.connected} label="Kafka metrics" />
          <StatusDot connected={dashboard.connected} label="Client frames" />
        </div>
      </div>
    </section>
  );
}
