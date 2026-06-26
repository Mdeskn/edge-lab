#!/usr/bin/env python3
"""
Experiment phase controller for the EdgeLab Network VM (172.22.174.148).

Watches a phase file written by SeQaM (or by an operator manually).
When the file content changes, it:
  1. Runs the corresponding tc_control.sh command to apply network shaping.
  2. Publishes the new phase and applied parameters to Kafka.

Designed to run as a plain Python script directly on the Network VM:

    Setup:
        cd /home/mae/network_load
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

    Run:
        python3 -u tc_controller.py \\
            --phase-file /tmp/edgelab_phase \\
            --kafka-bootstrap-servers 172.22.174.149:9092 \\
            --topic edgelab.phase \\
            --tc-script /home/mae/network_load/tc_control.sh

    Dry-run mode (no sudo, just logs + Kafka):
        python3 -u tc_controller.py --dry-run ...

    Manual phase tests:
        echo baseline      > /tmp/edgelab_phase
        echo cycle_start   > /tmp/edgelab_phase
        echo cycle_end     > /tmp/edgelab_phase
        echo bandwidth_200 > /tmp/edgelab_phase
        echo bandwidth_50  > /tmp/edgelab_phase
        echo bandwidth_20  > /tmp/edgelab_phase
        echo bandwidth_5   > /tmp/edgelab_phase
        echo jitter_light  > /tmp/edgelab_phase
        echo gpu_load      > /tmp/edgelab_phase
        echo mixed         > /tmp/edgelab_phase

        Numeric aliases also work:
        echo 0 > /tmp/edgelab_phase   (baseline)
        echo 2 > /tmp/edgelab_phase   (bandwidth_50)

    Clear state manually:
        sudo /home/mae/network_load/tc_control.sh clear

    Check current state:
        sudo /home/mae/network_load/tc_control.sh show

    Environment variables (all overridden by CLI args):
        KAFKA_BROKERS           default: 172.22.174.149:9092
        KAFKA_PHASE_TOPIC       default: edgelab.phase
        PHASE_FILE              default: /tmp/edgelab_phase
        PHASE_POLL_INTERVAL_SEC default: 1.0
        TC_SCRIPT               default: /home/mae/network_load/tc_control.sh
        COMMAND_TIMEOUT_SEC     default: 10
        PHASE_REPUBLISH_SEC     default: 5
        LOG_LEVEL               default: INFO

Phase timing:
    SeQaM controls the experiment phase timing externally by writing to PHASE_FILE.
    tc_controller.py applies a PERSISTENT tc state for each phase and leaves it
    active until the next phase change. There is no auto-expiry or duration limit.
    - "baseline": clears all tc rules (no shaping).
    - "gpu_load": also clears tc rules; GPU load is handled by the external
      GPU load client on 172.22.174.145, not by the Network VM.
    - All other phases: apply the corresponding tc_control.sh command without
      a duration argument, so rules remain active indefinitely.

Published Kafka message structure:
    {
        "timestamp":      float   Unix timestamp
        "source":         str     "network-vm"
        "phase":          str     e.g. "bandwidth_5"
        "phase_index":    int     numeric index (0-5)
        "description":    str     human-readable description
        "tc_command":     str     full sudo command that was (or would be) run
        "tc_parameters":  dict    mode, bandwidth, burst, tbf_latency, delay_ms,
                                  jitter_ms, packet_loss_percent, duration_seconds
        "status":         str     "applied" | "failed" | "dry_run"
        "error":          str     only present when status == "failed"
    }

Role split:
    tc_control.sh                    applies tc/iptables rules (already on Network VM)
    tc_controller.py (this)          watches phase file, triggers tc_control.sh, publishes phase
    network_conditions_publisher.py  reads live tc state, publishes actual network metrics

Kafka infrastructure:
    Kafka broker:  172.22.174.149:9092
    Kafka UI:      http://172.22.174.149:8080
    Topic:         edgelab.phase

Sudo requirement:
    The user running this script needs passwordless sudo for tc_control.sh:
        echo "mae ALL=(ALL) NOPASSWD: /home/mae/network_load/tc_control.sh" \\
            | sudo tee /etc/sudoers.d/edgelab-tc
"""

# ---------------------------------------------------------------------------
# Phase map: edit this section to add/change experiment phases.
# Keys are the canonical phase names written to PHASE_FILE.
# tc_args are passed directly to tc_control.sh after the script path.
# ---------------------------------------------------------------------------

#: Map from canonical phase name to phase configuration.
#: Edit the entries here to change the tc_control.sh commands for each phase.
#:
#: Design notes:
#:   - NO duration_seconds are passed to tc_control.sh. SeQaM controls the timing
#:     of experiment phases externally. tc_controller.py applies a persistent TC
#:     state when the phase changes and leaves it active until the next phase.
#:   - baseline and gpu_load both run "clear" because neither requires network shaping.
#:     GPU load is applied by the external GPU load client, not by the Network VM.
#:   - tc_parameters mirrors the arguments that were passed to tc_control.sh, for
#:     informational publishing to Kafka consumers (students reading edgelab.phase).
PHASE_MAP: dict[str, dict] = {
    "baseline": {
        "index": 0,
        "description": "No network shaping. Clear traffic control rules.",
        "tc_args": ["clear"],
        "tc_parameters": {
            "mode": "clear",
            "bandwidth": "unlimited",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "tbf_latency": None,
            "duration_seconds": None,
        },
    },
    "cycle_start": {
        "index": 8,
        "description": "Cycle boundary. Clear traffic control rules.",
        "tc_args": ["clear"],
        "tc_parameters": {
            "mode": "clear",
            "bandwidth": "unlimited",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "tbf_latency": None,
            "duration_seconds": None,
        },
    },
    "cycle_end": {
        "index": 9,
        "description": "Cycle finished. Clear traffic control rules.",
        "tc_args": ["clear"],
        "tc_parameters": {
            "mode": "clear",
            "bandwidth": "unlimited",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "tbf_latency": None,
            "duration_seconds": None,
        },
    },
    "bandwidth_200": {
        "index": 1,
        "description": "Limit bandwidth to 200mbit.",
        "tc_args": ["tbf", "200mbit", "2mbit", "50ms"],
        "tc_parameters": {
            "mode": "tbf",
            "bandwidth": "200mbit",
            "burst": "2mbit",
            "tbf_latency": "50ms",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "duration_seconds": None,
        },
    },
    "bandwidth_50": {
        "index": 2,
        "description": "Legacy bandwidth phase: limit bandwidth to 50mbit.",
        "tc_args": ["tbf", "50mbit", "2mbit", "50ms"],
        "tc_parameters": {
            "mode": "tbf",
            "bandwidth": "50mbit",
            "burst": "2mbit",
            "tbf_latency": "50ms",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "duration_seconds": None,
        },
    },
    "bandwidth_20": {
        "index": 7,
        "description": "Limit bandwidth to 20mbit for the JPEG remote inference path.",
        "tc_args": ["tbf", "20mbit", "512kb", "50ms"],
        "tc_parameters": {
            "mode": "tbf",
            "bandwidth": "20mbit",
            "burst": "512kb",
            "tbf_latency": "50ms",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "duration_seconds": None,
        },
    },
    "bandwidth_5": {
        "index": 6,
        "description": "Limit bandwidth to 5mbit for the JPEG remote inference path.",
        "tc_args": ["tbf", "5mbit", "256kb", "50ms"],
        "tc_parameters": {
            "mode": "tbf",
            "bandwidth": "5mbit",
            "burst": "256kb",
            "tbf_latency": "50ms",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "duration_seconds": None,
        },
    },
    "jitter_light": {
        "index": 3,
        "description": "Apply visible netem delay/jitter with high bandwidth.",
        "tc_args": ["netem_tbf", "25ms", "15ms", "1gbit", "2mbit", "50ms"],
        "tc_parameters": {
            "mode": "netem_tbf",
            "bandwidth": "1gbit",
            "burst": "2mbit",
            "tbf_latency": "50ms",
            "delay_ms": 25.0,
            "jitter_ms": 15.0,
            "packet_loss_percent": 0.0,
            "duration_seconds": None,
        },
    },
    "gpu_load": {
        "index": 4,
        "description": "GPU load phase. Network shaping is cleared; GPU load is handled externally.",
        "tc_args": ["clear"],
        "tc_parameters": {
            "mode": "clear",
            "bandwidth": "unlimited",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "tbf_latency": None,
            "duration_seconds": None,
        },
    },
    "mixed": {
        "index": 5,
        "description": "Mixed phase. Apply 20mbit shaping; GPU load is triggered externally.",
        "tc_args": ["tbf", "20mbit", "512kb", "50ms"],
        "tc_parameters": {
            "mode": "tbf",
            "bandwidth": "20mbit",
            "burst": "512kb",
            "tbf_latency": "50ms",
            "delay_ms": 0.0,
            "jitter_ms": 0.0,
            "packet_loss_percent": 0.0,
            "duration_seconds": None,
        },
    },
}

# Numeric aliases: "0" -> "baseline", "2" -> "bandwidth_50", etc.
_INDEX_ALIASES: dict[str, str] = {
    str(v["index"]): k for k, v in PHASE_MAP.items()
}

# ---------------------------------------------------------------------------

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase file reader
# ---------------------------------------------------------------------------

def read_phase(path: Path) -> str | None:
    """
    Read a canonical phase name from the watched file.

    Accepts both canonical names (e.g. "bandwidth_50") and numeric index
    aliases (e.g. "2"). Returns None if the file does not exist, is empty,
    or contains an unrecognised value.
    """
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Could not read phase file %s: %s", path, exc)
        return None

    if not raw:
        return None

    # Support numeric aliases
    name = _INDEX_ALIASES.get(raw, raw)

    if name not in PHASE_MAP:
        valid = ", ".join(
            f"{k} ({v['index']})" for k, v in PHASE_MAP.items()
        )
        logger.warning(
            "Unknown phase %r in %s -- valid phases: %s", raw, path, valid
        )
        return None

    return name


# ---------------------------------------------------------------------------
# tc_control.sh executor
# ---------------------------------------------------------------------------

def apply_phase(
    phase: str,
    tc_script: str,
    dry_run: bool,
    command_timeout_s: float = 10.0,
) -> tuple[str, str | None]:
    """
    Run tc_control.sh with the arguments for the given phase.

    No duration argument is passed to tc_control.sh. SeQaM controls
    experiment timing externally; tc rules stay active until the next
    phase change triggers a new tc_control.sh call.

    Returns (status, error_message_or_None) where status is one of:
        "applied"   command ran and exited 0
        "failed"    command exited non-zero or raised an exception
        "dry_run"   dry-run mode; command was logged but not executed
    """
    info = PHASE_MAP[phase]
    cmd = ["sudo", tc_script, *info["tc_args"]]
    cmd_str = " ".join(cmd)

    if dry_run:
        logger.info("[dry-run] Would run: %s", cmd_str)
        return "dry_run", None

    logger.info("Applying phase %r: %s", phase, cmd_str)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=command_timeout_s,
        )
        if result.returncode != 0:
            err = (result.stderr.strip() or result.stdout.strip())[:500]
            logger.error(
                "tc_control.sh exited %d for phase %r: %s",
                result.returncode,
                phase,
                err,
            )
            return "failed", err
        stdout = result.stdout.strip()
        if stdout:
            logger.debug("tc_control.sh output: %s", stdout)
        logger.info("Phase %r applied successfully", phase)
        return "applied", None

    except subprocess.TimeoutExpired:
        msg = f"tc_control.sh timed out after {command_timeout_s}s for phase {phase!r}"
        logger.error(msg)
        return "failed", msg

    except Exception as exc:
        msg = str(exc)
        logger.error("tc_control.sh error for phase %r: %s", phase, msg)
        return "failed", msg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch a phase file and apply tc_control.sh rules, publishing phase events to Kafka.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--phase-file",
        default=os.environ.get("PHASE_FILE", "/tmp/edgelab_phase"),
        metavar="PATH",
        help="Path to the phase file written by SeQaM or an operator",
    )
    p.add_argument(
        "--kafka-bootstrap-servers",
        default=os.environ.get("KAFKA_BROKERS", "172.22.174.149:9092"),
        metavar="BROKERS",
        help="Kafka bootstrap servers (host:port)",
    )
    p.add_argument(
        "--topic",
        default=os.environ.get("KAFKA_PHASE_TOPIC", "edgelab.phase"),
        help="Kafka topic to publish phase events to",
    )
    p.add_argument(
        "--tc-script",
        default=os.environ.get("TC_SCRIPT", "/home/mae/network_load/tc_control.sh"),
        metavar="PATH",
        help="Absolute path to tc_control.sh on the Network VM",
    )
    p.add_argument(
        "--poll-interval-s",
        type=float,
        default=float(os.environ.get("PHASE_POLL_INTERVAL_SEC", "1.0")),
        metavar="SECONDS",
        help="How often to re-read the phase file",
    )
    p.add_argument(
        "--command-timeout-s",
        type=float,
        default=float(os.environ.get("COMMAND_TIMEOUT_SEC", "10")),
        metavar="SECONDS",
        help="Timeout in seconds for each tc_control.sh subprocess call",
    )
    p.add_argument(
        "--republish-interval-s",
        type=float,
        default=float(os.environ.get("PHASE_REPUBLISH_SEC", "5")),
        metavar="SECONDS",
        help="Republish the current phase to Kafka at this interval, even when it has not changed",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log and publish what would happen without running sudo/tc_control.sh",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Watch the phase file, apply tc rules, and publish phase events to Kafka."""
    args = _parse_args()

    phase_file = Path(args.phase_file)
    tc_script = args.tc_script
    poll_interval = args.poll_interval_s
    command_timeout_s = args.command_timeout_s
    republish_interval_s = args.republish_interval_s
    dry_run = args.dry_run
    kafka_brokers = args.kafka_bootstrap_servers
    topic = args.topic

    producer = Producer(
        {
            "bootstrap.servers": kafka_brokers,
            "client.id": "edge-lab-tc-controller",
        }
    )

    logger.info("TC controller started")
    logger.info("  phase file : %s", phase_file)
    logger.info("  tc script  : %s", tc_script)
    logger.info("  Kafka      : %s -> %s", kafka_brokers, topic)
    logger.info("  poll       : %.2fs", poll_interval)
    logger.info("  cmd timeout: %.0fs", command_timeout_s)
    logger.info("  republish : %.0fs", republish_interval_s)
    if dry_run:
        logger.info("  mode       : DRY RUN (sudo/tc commands will NOT be executed)")

    def _delivery_report(err, msg) -> None:
        if err:
            logger.error("Kafka delivery failed: %s", err)

    last_phase: str | None = None
    last_phase_publish_time = 0.0
    last_phase_payload: dict | None = None
    phase_file_warned = False

    while True:
        if not phase_file.exists():
            if not phase_file_warned:
                logger.info("Waiting for phase file %s ...", phase_file)
                phase_file_warned = True
            time.sleep(poll_interval)
            continue

        phase_file_warned = False
        phase = read_phase(phase_file)

        if phase is None:
            time.sleep(poll_interval)
            continue

        now = time.time()

        if phase == last_phase:
            # Republish current phase periodically so dashboards that start late
            # still learn the active experiment phase.
            if last_phase_payload is not None and now - last_phase_publish_time >= republish_interval_s:
                try:
                    producer.produce(
                        topic,
                        key="phase",
                        value=json.dumps(last_phase_payload).encode("utf-8"),
                        callback=_delivery_report,
                    )
                    producer.poll(0)
                    last_phase_publish_time = now
                    logger.info(
                        "Republished phase %r (index=%d) to %s",
                        phase,
                        last_phase_payload["phase_index"],
                        topic,
                    )
                except Exception as exc:
                    logger.error("Kafka republish error: %s", exc)
            time.sleep(poll_interval)
            continue

        # Phase has changed - apply and publish.
        info = PHASE_MAP[phase]
        cmd_str = "sudo " + tc_script + " " + " ".join(info["tc_args"])

        status, error = apply_phase(phase, tc_script, dry_run, command_timeout_s)

        payload: dict = {
            "timestamp": time.time(),
            "source": "network-vm",
            "phase": phase,
            "phase_index": info["index"],
            "description": info["description"],
            "tc_command": cmd_str,
            "tc_parameters": info["tc_parameters"],
            "status": status,
        }
        if error:
            payload["error"] = error

        try:
            producer.produce(
                topic,
                key="phase",
                value=json.dumps(payload).encode("utf-8"),
                callback=_delivery_report,
            )
            producer.poll(0)
            last_phase_payload = payload
            last_phase_publish_time = time.time()
            logger.info(
                "Published phase %r (index=%d, status=%s) to %s",
                phase,
                info["index"],
                status,
                topic,
            )
        except Exception as exc:
            logger.error("Kafka produce error: %s", exc)

        # Track last seen phase regardless of tc success so we don't spam
        # the same command on every poll tick after a transient failure.
        last_phase = phase
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
