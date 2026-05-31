# SeQaM Scenario

`scenario.json` defines the 4-phase load loop used during the 2-minute experiment run.
SeQaM executes the load commands over SSH. The Network VM phase sidecar watches
`/tmp/edgelab_phase` and publishes phase changes to
`/edgelab/server/events/phase`.

## Phases

| Time (s) | Phase | GPU load | Network load |
|----------|-------|----------|--------------|
| 0–30     | `baseline`     | No  | No  |
| 30–60    | `gpu_load`     | Yes | No  |
| 60–90    | `network_load` | No  | Yes |
| 90–120   | `combined`     | Yes | Yes |

To run the scenario continuously, use the provided HTTP wrapper on a host that
can reach the SeQaM API:

```bash
SEQAM_API_URL=http://<seqam-host>:8000 \
./scripts/run_scenario_loop.sh
```

The wrapper posts `scenario.json` to SeQaM's
`POST /config/ExperimentConfig.json` endpoint. Because the scenario sets
`execute_immediately` to `true`, each POST starts one experiment run.

## Configure SeQaM SSH targets

Merge the entries from `ScenarioConfig.json` into the SeQaM host's
`~/.seqam_fh_dortmund_project_emulate/ScenarioConfig.json` before starting
SeQaM. Both machines intentionally appear under `router`: this SeQaM revision
loads `router` entries as static SSH targets, while `server` entries are
resolved from live component registrations.

Install SeQaM's generated `ecdsa.pub` key for user `mae` on both machines and
place the scripts under `/scripts`. On the Network VM, start
`docker-compose.netvm.yml` so `tc_controller.py` can bridge the phase file to
Kafka.

The explicit `command:'...'` part of each SSH action is required by the current
SeQaM parser to preserve the trailing shell command.
