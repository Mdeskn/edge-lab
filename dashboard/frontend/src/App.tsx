import { Activity, RadioTower } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchDashboardState, subscribeToDashboard } from "./api";
import { Charts } from "./components/Charts";
import { InfrastructurePanel } from "./components/InfrastructurePanel";
import { InterpretationPanel } from "./components/InterpretationPanel";
import { MetricsCards } from "./components/MetricsCards";
import { StatusDot } from "./components/StatusDot";
import { SummaryPanel } from "./components/SummaryPanel";
import { VideoPanel } from "./components/VideoPanel";
import type { DashboardState } from "./types";

const phaseDescriptions: Record<string, string> = {
  baseline: "No artificial load",
  network_load: "Network delay, jitter, or packet loss active",
  gpu_load: "GPU and Triton server under load",
  combined: "Network and server load active",
  unknown: "Waiting for phase metrics",
};

export default function App() {
  const [state, setState] = useState<DashboardState | null>(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setState(null);
    setError(null);
    fetchDashboardState()
      .then((next) => active && setState(next))
      .catch((reason: Error) => active && setError(reason.message));
    const unsubscribe = subscribeToDashboard(
      (next) => {
        setState(next);
        setError(null);
      },
      setSocketConnected,
    );
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  const phase = state?.experiment_phase || "unknown";

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <Activity size={28} />
          <div>
            <h1>Edge-Lab Live Dashboard</h1>
            <p>Service placement and tracking quality</p>
          </div>
        </div>
        <div className="topbar-controls">
          <StatusDot connected={socketConnected} label={socketConnected ? "Live" : "Reconnecting"} />
          <div className="phase-block">
            <span className="eyebrow">Experiment phase</span>
            <strong>{phase.replace("_", " ")}</strong>
            <small>{phaseDescriptions[phase] || phaseDescriptions.unknown}</small>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <RadioTower size={18} />
          {error}. The dashboard will reconnect automatically.
        </div>
      )}

      {state ? (
        <div className="dashboard-shell">
          <section className="overview-grid">
            <VideoPanel state={state} />
            <MetricsCards state={state} />
          </section>
          <section className="signal-grid">
            <InfrastructurePanel state={state} />
            <InterpretationPanel state={state} />
          </section>
          <Charts state={state} />
          <SummaryPanel state={state} />
        </div>
      ) : (
        <div className="loading-state">Connecting to Edge-Lab dashboard...</div>
      )}
    </main>
  );
}
