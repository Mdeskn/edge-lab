# SeQaM Scenario

`scenario.json` defines the 4-phase load loop used during the 2-minute experiment run.

## Phases

| Time (s) | Phase | GPU load | Network load |
|----------|-------|----------|--------------|
| 0–30     | `baseline`     | No  | No  |
| 30–60    | `gpu_load`     | Yes | No  |
| 60–90    | `network_load` | No  | Yes |
| 90–120   | `combined`     | Yes | Yes |

SeQaM runs the scenario once and then exits (the `exit` command at t=120s
terminates the process). To run it continuously, use the provided wrapper:

```bash
./scripts/run_scenario_loop.sh
```

SeQaM publishes the current phase name to the `experiment.phase` Kafka topic
so the SP-Agent can react.

## Loading into SeQaM

Upload `scenario.json` through the SeQaM web interface or point the SeQaM
CLI at this file. Ensure the `experiment.phase` topic exists before starting.
