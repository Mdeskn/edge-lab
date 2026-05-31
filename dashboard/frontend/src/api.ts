import type { DashboardState } from "./types";

export const API_BASE = (import.meta.env.VITE_API_URL || "http://localhost:8080").replace(/\/$/, "");

export function resolveApiUrl(path?: string | null): string | null {
  return path ? `${API_BASE}${path}` : null;
}

export async function fetchDashboardState(groupId: string): Promise<DashboardState> {
  const response = await fetch(`${API_BASE}/api/state?group_id=${encodeURIComponent(groupId)}`);
  if (!response.ok) {
    throw new Error(`Dashboard API returned ${response.status}`);
  }
  return response.json();
}

export function subscribeToDashboard(
  groupId: string,
  onUpdate: (state: DashboardState) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  let stopped = false;
  let socket: WebSocket | null = null;
  let retryTimer: number | undefined;
  const websocketBase = API_BASE.replace(/^http/, "ws");

  const connect = () => {
    socket = new WebSocket(`${websocketBase}/ws?group_id=${encodeURIComponent(groupId)}`);
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
