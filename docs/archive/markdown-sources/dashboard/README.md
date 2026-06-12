# Edge-Lab Live Dashboard

The dashboard is optional and runs on each student group's Raspberry Pi beside
the client. The experiment pipeline continues to run when the dashboard is
disabled, offline, or missing its dependencies.

For a feature-by-feature explanation and exact data flow, see
[FEATURES_AND_ARCHITECTURE.md](FEATURES_AND_ARCHITECTURE.md).

## Architecture

- The Pi client sends throttled, JPEG-compressed annotated frames to the backend.
- The backend consumes scored frames and infrastructure metrics from Kafka when
  Kafka is configured.
- The backend keeps bounded in-memory rolling histories and broadcasts updates
  over WebSocket.
- The React frontend reconnects automatically and shows `N/A` for metrics that
  have not arrived yet.

The frame POST also includes the scored frame metrics. Kafka remains the source
of infrastructure metrics and phase updates during the full lab experiment.

## Configure

Copy the root environment template and edit values for the machine that runs
each component:

```bash
cp .env.example .env
```

Dashboard backend:

```dotenv
GROUP_ID=1
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
# Optional override. Leave blank to derive http://<pi-host>:8080 in the browser.
DASHBOARD_PUBLIC_API_URL=
KAFKA_BROKERS=
# Optional override; defaults to dnn_partition.client_metrics.
APP_METRICS_TOPIC=
KAFKA_GPU_TOPIC=dnn_partition.server_metrics
KAFKA_NET_TOPIC=edgelab.network.metrics
KAFKA_PHASE_TOPIC=edgelab.phase
```

Pi client:

```dotenv
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://localhost:8080
DASHBOARD_FPS=5
DASHBOARD_JPEG_QUALITY=70
DASHBOARD_FRAME_WIDTH=960
```

Keep `DASHBOARD_URL=http://localhost:8080`: the client and dashboard backend run
on the same Pi. The browser frontend derives `http://<pi-host>:8080`
automatically. Set `DASHBOARD_PUBLIC_API_URL` only when an explicit override is
needed.

## Start With Docker Compose

Create `.env` in the repository root, then start the Pi client and dashboard:

```bash
docker compose -f docker-compose.pi.yml up --build
```

Open `http://PI_IP:5173`. The backend health endpoint is
`http://PI_IP:8080/health`.

For dashboard-only troubleshooting on the same Pi, run:

```bash
docker compose -f dashboard/docker-compose.yml up --build
```

## Run The Experiment

Full Kafka and Triton lab:

1. Configure `KAFKA_BROKERS`, `TRITON_URL`, and the Kafka topic variables.
2. Start Triton and the GPU metrics publisher with
   `docker compose -f docker-compose.gpu-server.yml up --build`.
3. Start the network publisher with
   `docker compose -f docker-compose.netvm.yml up --build`.
4. Start the Pi client and dashboard with `docker-compose.pi.yml`.

## Backend API

- `GET /health`
- `GET /api/state`
- `GET /api/history`
- `POST /api/frame`
- `GET /api/frame`
- `POST /api/reset`
- `WS /ws`

## Troubleshooting

### The browser says reconnecting

Check `http://PI_IP:8080/health`. If it does not load, start the backend or
check whether port `8080` is already in use.

### Video is missing but Kafka charts update

Verify `DASHBOARD_ENABLED=true` on the Pi and make sure `DASHBOARD_URL` is
reachable from the Pi. The client logs `DashboardPublisher enabled` at startup.

### GPU or network values show N/A

Check `KAFKA_BROKERS`, topic names, and the GPU and network publisher logs.

### The dashboard is too heavy for the Pi

Lower `DASHBOARD_FPS`, `DASHBOARD_FRAME_WIDTH`, or `DASHBOARD_JPEG_QUALITY`.
The publisher drops stale snapshots automatically instead of delaying inference.
