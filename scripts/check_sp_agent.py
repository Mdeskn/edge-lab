#!/usr/bin/env python3
"""
Check that an SP-Agent submission loads and behaves, without the lab hardware.

A broken decide() used to produce a run that looked complete: the base class
caught the exception, kept the previous mode, and the group scored like the
always-local baseline. That is only discoverable after booking Pi, GPU server,
and network shaper time. This runs the same import path and calls decide()
against synthetic metric snapshots in under a second.

    python scripts/check_sp_agent.py                # checks student.sp_agent
    python scripts/check_sp_agent.py --agent example

Exit status is 0 when the agent is usable and 1 otherwise, so it can gate a
submission in CI or a pre-flight check on the Pi.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "client"))

# The agent must not need real data files or a broker to be checked.
os.environ.setdefault("VIDEO_PATH", "/nonexistent/video.mp4")
os.environ.setdefault("GROUND_TRUTH_PATH", "/nonexistent/gt.csv")
os.environ.setdefault("MODEL_PATH", "/nonexistent/model.onnx")
os.environ.setdefault("KAFKA_BROKERS", "")
os.environ.setdefault("DASHBOARD_ENABLED", "false")

VALID_PLACEMENTS = ("local", "remote")

#: How many times decide() is called per scenario. Strategies commonly use
#: hysteresis (the reference one requires the same new decision twice before
#: switching), so a single call would report a settling value rather than what
#: the agent actually converges to while a phase holds.
SETTLE_CALLS = 4

#: Metric snapshots the agent must survive. The first is the state every agent
#: actually starts in: no Kafka message has arrived yet, so every dict is empty
#: and every probe field is None. Agents that index instead of using .get()
#: fail here rather than on the Pi.
SCENARIOS: list[tuple[str, dict]] = [
    ("cold start, no metrics yet", {}),
    (
        "clean baseline",
        {
            "gpu": {"gpu_util_pct": 5.0, "yolo_queue_ms": 1.0, "total_pending": 0},
            "net": {"delay_ms": 1.0, "jitter_ms": 0.2, "packet_loss_pct": 0.0,
                    "bandwidth": "1gbit"},
            "phase": "cycle_start",
            "latencies": [45.0] * 5,
            "probe": (40.0, "ok"),
        },
    ),
    (
        "GPU saturated",
        {
            "gpu": {"gpu_util_pct": 99.0, "yolo_queue_ms": 250.0, "total_pending": 40},
            "net": {"delay_ms": 1.0, "jitter_ms": 0.2, "packet_loss_pct": 0.0,
                    "bandwidth": "1gbit"},
            "phase": "gpu_load",
            "latencies": [600.0] * 5,
            "probe": (600.0, "ok"),
        },
    ),
    (
        "network degraded",
        {
            "gpu": {"gpu_util_pct": 10.0, "yolo_queue_ms": 2.0, "total_pending": 0},
            "net": {"delay_ms": 120.0, "jitter_ms": 45.0, "packet_loss_pct": 5.0,
                    "bandwidth": "20mbit"},
            "phase": "mixed",
            "latencies": [800.0] * 5,
            "probe": (None, "failed"),
        },
    ),
    (
        "remote unreachable",
        {
            "gpu": {},
            "net": {"delay_ms": 0.0, "jitter_ms": 0.0, "packet_loss_pct": 100.0,
                    "bandwidth": "unknown"},
            "phase": "bandwidth_20",
            "latencies": [],
            "probe": (None, "unavailable"),
        },
    ),
]


def build_agent(agent_name: str):
    """Load and construct the agent exactly the way main.py does."""
    import importlib

    from config import load_config
    from shared_state import SharedState

    os.environ["SP_AGENT_CLASS"] = agent_name
    config = load_config()
    shared_state = SharedState(initial_mode=config.initial_processing_mode)

    module_name = (
        "student.sp_agent" if agent_name == "student" else "student.sp_agent_example"
    )
    module = importlib.import_module(module_name)
    agent_class = module.SPAgent
    return agent_class(config, shared_state), shared_state, agent_class


def apply_scenario(agent, scenario: dict) -> None:
    """Populate the agent's private metric state for one synthetic snapshot."""
    import time
    from collections import deque

    with agent._metrics_lock:
        agent._gpu_metrics = dict(scenario.get("gpu", {}))
        if "net" in scenario:
            net = dict(scenario["net"])
            net.setdefault("packet_loss_percent", net.get("packet_loss_pct", 0.0))
            agent._net_metrics = net
        agent._experiment_phase = scenario.get("phase", "cycle_start")
        latencies = scenario.get("latencies", [])
        agent._recent_latencies = deque(latencies, maxlen=20)
        agent._recent_remote_latencies = deque(latencies, maxlen=20)
        probe_latency, probe_status = scenario.get("probe", (None, "unavailable"))
        agent._last_remote_probe_latency = probe_latency
        agent._last_remote_probe_status = probe_status
        agent._last_remote_probe_at = time.time() if probe_status != "unavailable" else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default="student",
        choices=("student", "example"),
        help="which submission to check (default: student)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show the agent's own log output"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s: %(message)s",
    )

    print(f"Checking SP-Agent: {args.agent}")

    try:
        agent, _shared_state, agent_class = build_agent(args.agent)
    except Exception as exc:
        print(f"  FAIL  could not construct the agent: {type(exc).__name__}: {exc}")
        print("\nThe agent could not even be loaded. Check for syntax errors and "
              "that class SPAgent still subclasses SPAgentBase.")
        return 1

    print(f"  ok    loaded {agent_class.__name__} from client/student/")

    from student.sp_agent_base import SPAgentBase

    if type(agent).decide is SPAgentBase.decide:
        print("  FAIL  decide() is not overridden: the agent would never choose "
              "a placement")
        return 1
    print("  ok    decide() is overridden")

    failures = 0
    decisions: dict[str, int] = {}

    for label, scenario in SCENARIOS:
        apply_scenario(agent, scenario)
        decision = None
        failed = False

        for _ in range(SETTLE_CALLS):
            try:
                decision = agent.decide()
            except NotImplementedError:
                print(f"  FAIL  [{label}] decide() is still the unimplemented stub")
                failed = True
                break
            except Exception as exc:
                print(f"  FAIL  [{label}] decide() raised {type(exc).__name__}: {exc}")
                failed = True
                break

            if decision not in VALID_PLACEMENTS:
                print(
                    f"  FAIL  [{label}] decide() returned {decision!r}; "
                    f"it must return one of {VALID_PLACEMENTS}"
                )
                failed = True
                break

            # The run loop applies each decision, so the next call sees the
            # placement its own previous return produced.
            agent.set_mode(decision)

        if failed:
            failures += 1
            continue

        decisions[decision] = decisions.get(decision, 0) + 1
        print(f"  ok    [{label}] -> {decision}")

    if failures:
        print(f"\n{failures} of {len(SCENARIOS)} scenarios failed. "
              "Fix these before running on the Pi.")
        return 1

    print(f"\nAll {len(SCENARIOS)} scenarios passed.")
    if len(decisions) == 1:
        only = next(iter(decisions))
        print(
            f"Note: the agent chose {only!r} in every scenario. That is valid "
            "and it will run, but a strategy that never reacts to the metrics "
            "scores like the fixed baseline."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
