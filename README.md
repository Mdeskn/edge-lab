# Edge-Lab Pi App

This branch contains only the application that runs on the Raspberry Pi:

- `client/` - video processing, local YOLO, remote inference client, scoring, Kafka metrics
- `dashboard/` - live dashboard backend and frontend
- `data/` - model, video, ground truth, and saved result output
- `seqam/ExperimentConfig.json` - phase timing file mounted by the client to derive cycle duration
- `docker-compose.yml` - Pi runtime stack

Non-Pi VM scripts, GPU-server gateway files, network-VM publishers, SeQaM helper files,
and long-form lab documentation were moved into the local ignored folder:

```text
distributed-materials/
```

That folder is intentionally not tracked on this final Pi-app branch.

## Run On The Pi

Create `.env` from `.env.example`, then start the app:

```bash
cp .env.example .env
docker compose up -d --build
```

Open the dashboard:

```text
http://<pi-host>:5173
```

Useful checks:

```bash
docker compose ps
docker compose logs -f client
docker compose logs -f dashboard-backend
```

To stop:

```bash
docker compose down
```
