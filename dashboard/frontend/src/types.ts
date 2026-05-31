export type MetricValue = number | string | null | undefined;

export interface FrameMetric {
  timestamp?: number;
  frame_number?: number;
  group_id?: string;
  experiment_phase?: string;
  processing_mode?: string;
  latency_ms?: number;
  displacement_px?: number;
  cumulative_displacement_px?: number;
}

export interface InfrastructureMetric {
  timestamp?: number;
  [key: string]: MetricValue;
}

export interface DashboardState {
  group_id: string;
  groups: string[];
  latest: FrameMetric;
  experiment_phase: string;
  frame: {
    sequence: number;
    frame_number?: number;
    url?: string | null;
    updated_at?: number | null;
  };
  latency: {
    rolling_average_ms?: number | null;
    min_ms?: number | null;
    max_ms?: number | null;
    p95_ms?: number | null;
  };
  displacement: {
    rolling_average_px?: number | null;
  };
  infrastructure: {
    gpu: InfrastructureMetric;
    network: InfrastructureMetric;
    kafka: {
      connected: boolean;
      detail: string;
      last_message_at?: number | null;
    };
    dashboard: {
      connected: boolean;
      detail: string;
      last_frame_at?: number | null;
    };
  };
  summary: {
    total_frames: number;
    local_frames: number;
    remote_frames: number;
    local_percentage: number;
    remote_percentage: number;
    average_latency_ms?: number | null;
    average_displacement_px?: number | null;
    cumulative_displacement_px: number;
    best_latency_ms?: number | null;
    worst_latency_ms?: number | null;
    best_displacement_px?: number | null;
    worst_displacement_px?: number | null;
  };
  history: {
    frames: FrameMetric[];
    gpu: InfrastructureMetric[];
    network: InfrastructureMetric[];
  };
  updated_at: number;
}
