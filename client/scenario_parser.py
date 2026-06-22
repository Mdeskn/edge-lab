"""
Parses a SeQaM scenario file to derive the cycle duration in seconds.

The cycle duration is the time between the first 'baseline' phase event and the
next 'baseline' phase event (one full rotation through all phases). If the
scenario only contains one baseline event, the duration is the timestamp of the
final event in the file.

This lets the app and dashboard adapt automatically when the scenario changes,
without requiring anyone to update a separate CYCLE_DURATION_SEC env var.
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default if the scenario file cannot be parsed or is unavailable.
DEFAULT_CYCLE_DURATION_SEC: float = 75.0

# Match commands like "... set_phase.sh PHASENAME ..." used in scenario events.
_SET_PHASE_RE = re.compile(r"set_phase\.sh\s+(\w+)")


def load_cycle_duration_sec(scenario_path: str | Path) -> float:
    """
    Read the scenario file and return the cycle duration in seconds.

    Returns DEFAULT_CYCLE_DURATION_SEC on any parse or IO error.
    The caller logs an informational message; the function itself logs
    a warning when falling back to the default.
    """
    path = Path(scenario_path)
    if not path.is_file():
        logger.warning(
            "Scenario file not found at %s; using default cycle duration %.0fs",
            path, DEFAULT_CYCLE_DURATION_SEC,
        )
        return DEFAULT_CYCLE_DURATION_SEC

    try:
        with open(path, "r") as fh:
            scenario = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not parse scenario %s (%s); using default cycle duration %.0fs",
            path, exc, DEFAULT_CYCLE_DURATION_SEC,
        )
        return DEFAULT_CYCLE_DURATION_SEC

    events = (
        scenario.get("events")
        or scenario.get("steps")
        or scenario.get("eventList")
        or []
    )
    if not isinstance(events, list) or not events:
        logger.warning(
            "Scenario %s has no events; using default cycle duration %.0fs",
            path, DEFAULT_CYCLE_DURATION_SEC,
        )
        return DEFAULT_CYCLE_DURATION_SEC

    baseline_times: list[float] = []
    last_event_time: float = 0.0

    for event in events:
        if not isinstance(event, dict):
            continue
        # Support both second-based "time" and "timestamp" keys; prefer
        # "time". SeQaM's native "executionTime" is in milliseconds.
        t_raw = event.get("time", event.get("timestamp"))
        time_scale = 1.0
        if t_raw is None:
            t_raw = event.get("executionTime")
            time_scale = 0.001
        try:
            t = float(t_raw) * time_scale
        except (TypeError, ValueError):
            continue
        last_event_time = max(last_event_time, t)

        cmd = event.get("command") or ""
        match = _SET_PHASE_RE.search(cmd)
        if match and match.group(1) == "baseline":
            baseline_times.append(t)

    # Two baselines means a complete cycle (start + return). Duration is delta.
    if len(baseline_times) >= 2:
        baseline_times.sort()
        duration = baseline_times[1] - baseline_times[0]
        if duration > 0:
            logger.info(
                "Cycle duration from scenario %s: %.1fs (baseline-to-baseline)",
                path, duration,
            )
            return duration

    # Fallback: last event timestamp as the duration. Better than the default
    # because it still reflects what's in the file.
    if last_event_time > 0:
        logger.info(
            "Cycle duration from scenario %s: %.1fs (last-event-time fallback)",
            path, last_event_time,
        )
        return last_event_time

    logger.warning(
        "Could not derive cycle duration from %s; using default %.0fs",
        path, DEFAULT_CYCLE_DURATION_SEC,
    )
    return DEFAULT_CYCLE_DURATION_SEC
