# Edge-Lab Live Dashboard

The dashboard is optional. The experiment pipeline continues to run when the
dashboard is disabled, offline, or missing its dependencies.

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
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
DASHBOARD_PUBLIC_API_URL=http://DASHBOARD_HOST:8080
KAFKA_BROKERS=
APP_METRICS_TOPICS=/edgelab/app/metrics/group1,/edgelab/app/metrics/group2,/edgelab/app/metrics/group3,/edgelab/app/metrics/group4
KAFKA_GPU_TOPIC=/edgelab/server/metrics
KAFKA_NET_TOPIC=/edgelab/network/metrics
KAFKA_PHASE_TOPIC=/edgelab/server/events/phase
```

Pi client:

```dotenv
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://DASHBOARD_HOST:8080
DASHBOARD_FPS=5
DASHBOARD_JPEG_QUALITY=70
DASHBOARD_FRAME_WIDTH=960
```

Use the dashboard computer's reachable IP address for `DASHBOARD_HOST` in the
Pi client's `DASHBOARD_URL` and in `DASHBOARD_PUBLIC_API_URL`. Do not use
`localhost` unless the client, browser, and dashboard backend run on the same
computer.

## Start With Docker Compose

Create `.env` in the repository root, then run:

```bash
docker compose -f dashboard/docker-compose.yml up --build
```

Open `http://DASHBOARD_HOST:5173`. The backend health endpoint is
`http://DASHBOARD_HOST:8080/health`.

## Run The Experiment

Full Kafka and Triton lab:

1. Configure `KAFKA_BROKERS`, `TRITON_URL`, and the Kafka topic variables.
2. Start Triton and the GPU metrics publisher with
   `docker compose -f docker-compose.gpu-server.yml up --build`.
3. Start the network publisher with
   `docker compose -f docker-compose.netvm.yml up --build`.
4. Start the dashboard and Pi client.

## Backend API

- `GET /health`
- `GET /api/state?group_id=group1`
- `GET /api/history?group_id=group1`
- `POST /api/frame/group1`
- `GET /api/frame/group1`
- `WS /ws?group_id=group1`

## Troubleshooting

### The browser says reconnecting

Check `http://DASHBOARD_HOST:8080/health`. If it does not load, start the backend or
check whether port `8080` is already in use.

### Video is missing but Kafka charts update

Verify `DASHBOARD_ENABLED=true` on the Pi and make sure `DASHBOARD_URL` is
reachable from the Pi. The client logs `DashboardPublisher enabled` at startup.

### GPU or network values show N/A

Check `KAFKA_BROKERS`, topic names, and the GPU and network publisher logs.

### The dashboard is too heavy for the Pi

Lower `DASHBOARD_FPS`, `DASHBOARD_FRAME_WIDTH`, or `DASHBOARD_JPEG_QUALITY`.
The publisher drops stale snapshots automatically instead of delaying inference.
