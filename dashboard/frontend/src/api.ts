import type { DashboardState } from "./types";

const defaultApiUrl = `${window.location.protocol}//${window.location.hostname}:8080`;

export const API_BASE = (import.meta.env.VITE_API_URL || defaultApiUrl).replace(/\/$/, "");

export function resolveApiUrl(path?: string | null): string | null {
  return path ? `${API_BASE}${path}` : null;
}

export async function fetchDashboardState(): Promise<DashboardState> {
  const response = await fetch(`${API_BASE}/api/state`);
  if (!response.ok) {
    throw new Error(`Dashboard API returned ${response.status}`);
  }
  return response.json();
}

export async function setPlacementMode(mode: "local" | "remote"): Promise<void> {
  const response = await fetch(`${API_BASE}/api/placement`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Dashboard API returned ${response.status}`);
  }
}

export function subscribeToDashboard(
  onUpdate: (state: DashboardState) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  let stopped = false;
  let socket: WebSocket | null = null;
  let retryTimer: number | undefined;
  const websocketBase = API_BASE.replace(/^http/, "ws");

  const connect = () => {
    socket = new WebSocket(`${websocketBase}/ws`);
    socket.onopen = () => onConnection(true);
    socket.onmessage = (event) => {
      try {
        onUpdate(JSON.parse(event.data) as DashboardState);
      } catch {
        onConnection(false);
      }
    };
    socket.onerror = () => socket?.close();
    socket.onclose = () => {
      onConnection(false);
      if (!stopped) {
        retryTimer = window.setTimeout(connect, 1500);
      }
    };
  };

  connect();
  return () => {
    stopped = true;
    if (retryTimer) window.clearTimeout(retryTimer);
    socket?.close();
  };
}
