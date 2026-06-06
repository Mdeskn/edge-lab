# SeQaM Scenario

`scenario.json` defines the 4-phase load loop used during the 2-minute experiment run.
SeQaM executes the load commands over SSH. The Network VM phase controller (`tc_controller.py`)
watches `/tmp/edgelab_phase` and publishes phase changes to the `edgelab.phase` Kafka topic.

## Phases

| Time (s) | Phase | tc rule | GPU load |
|----------|-------|---------|----------|
| 0-30     | `baseline`     | clear (no shaping) | No  |
| 30-60    | `gpu_load`     | clear (no shaping) | Yes |
| 60-90    | `bandwidth_50` | tbf 50mbit cap     | No  |
| 90-120   | `mixed`        | tbf 50mbit cap     | Yes |

Phase names in `scenario.json` must exactly match entries in `tc_controller.py`'s `PHASE_MAP`.
The tc_controller reads the phase file and decides which `tc_control.sh` command to run based on that name.
SeQaM does not call `tc_apply.sh` or `tc_clear.sh` directly; all tc management goes through tc_controller.

## Running the scenario continuously

```bash
SEQAM_API_URL=http://172.22.174.149:8000 \
./scripts/run_scenario_loop.sh
```

The wrapper posts `scenario.json` to SeQaM's `POST /config/ExperimentConfig.json` endpoint.
Because the scenario sets `execute_immediately` to `true`, each POST starts one experiment run.

## Configure SeQaM SSH targets

Merge the entries from `ScenarioConfig.json` into the SeQaM host's
`~/.seqam_fh_dortmund_project_emulate/ScenarioConfig.json` before starting
SeQaM. Both machines intentionally appear under `router`: this SeQaM revision
loads `router` entries as static SSH targets, while `server` entries are
resolved from live component registrations.

Install SeQaM's generated `ecdsa.pub` key for user `mae` on both machines and
place the scripts under `/scripts`. On the Network VM, start
`tc_controller.py` so it can bridge the phase file to Kafka.

The explicit `command:'...'` part of each SSH action is required by the current
SeQaM parser to preserve the trailing shell command.
