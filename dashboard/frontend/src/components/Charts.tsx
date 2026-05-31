import type { DashboardState, FrameMetric, InfrastructureMetric } from "../types";

interface Series {
  label: string;
  color: string;
  values: Array<number | null>;
}

interface ChartProps {
  title: string;
  unit: string;
  series: Series[];
  min?: number;
  max?: number;
}

function SparkChart({ title, unit, series, min, max }: ChartProps) {
  const width = 420;
  const height = 150;
  const pad = 12;
  const values = series.flatMap((item) => item.values).filter((value): value is number => value != null);
  const yMin = min ?? (values.length ? Math.min(...values) : 0);
  const yMax = max ?? (values.length ? Math.max(...values) : 1);
  const span = Math.max(yMax - yMin, 1);
  const longest = Math.max(...series.map((item) => item.values.length), 1);
  const point = (value: number, index: number) => {
    const x = pad + index / Math.max(longest - 1, 1) * (width - pad * 2);
    const y = height - pad - (value - yMin) / span * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };

  return (
    <article className="chart-panel">
      <div className="chart-heading">
        <h3>{title}</h3>
        <div className="chart-legend">
          {series.map((item) => <span key={item.label}><i style={{ backgroundColor: item.color }} />{item.label}</span>)}
        </div>
      </div>
      {values.length ? (
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} over time`}>
          <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} className="chart-axis" />
          <line x1={pad} x2={width - pad} y1={pad} y2={pad} className="chart-gridline" />
          {series.map((item) => {
            const points = item.values
              .map((value, index) => value == null ? null : point(value, index))
              .filter(Boolean)
              .join(" ");
            return <polyline key={item.label} points={points} fill="none" stroke={item.color} strokeWidth="3" />;
          })}
          <text x={pad} y={pad + 11}>{yMax.toFixed(1)}{unit}</text>
          <text x={pad} y={height - pad - 5}>{yMin.toFixed(1)}{unit}</text>
        </svg>
      ) : (
        <div className="chart-empty">Waiting for metrics</div>
      )}
    </article>
  );
}

const frameValues = (frames: FrameMetric[], key: keyof FrameMetric) =>
  frames.map((frame) => typeof frame[key] === "number" ? frame[key] as number : null);

const infraValues = (items: InfrastructureMetric[], key: string) =>
  items.map((item) => typeof item[key] === "number" ? item[key] as number : null);

interface Props {
  state: DashboardState;
}

export function Charts({ state }: Props) {
  const { frames, gpu, network } = state.history;
  return (
    <section className="charts-section">
      <div className="section-heading">
        <span className="eyebrow">Rolling window</span>
        <h2>Live Trends</h2>
      </div>
      <div className="charts-grid">
        <SparkChart title="Latency" unit=" ms" series={[{ label: "latency", color: "#147d92", values: frameValues(frames, "latency_ms") }]} />
        <SparkChart title="Frame displacement" unit=" px" series={[{ label: "distance", color: "#d14a45", values: frameValues(frames, "displacement_px") }]} />
        <SparkChart title="Cumulative displacement" unit=" px" series={[{ label: "score", color: "#b37916", values: frameValues(frames, "cumulative_displacement_px") }]} />
        <SparkChart title="Placement mode" unit="" min={0} max={1} series={[{
          label: "remote = 1",
          color: "#6559a8",
          values: frames.map((frame) => frame.processing_mode === "remote" ? 1 : frame.processing_mode ? 0 : null),
        }]} />
        <SparkChart title="GPU utilization" unit="%" min={0} max={100} series={[{ label: "GPU", color: "#26845a", values: infraValues(gpu, "gpu_utilization_pct") }]} />
        <SparkChart title="Network conditions" unit=" ms" series={[
          { label: "delay", color: "#147d92", values: infraValues(network, "delay_ms") },
          { label: "jitter", color: "#b37916", values: infraValues(network, "jitter_ms") },
        ]} />
      </div>
    </section>
  );
}
