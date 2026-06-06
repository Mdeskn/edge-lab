#!/usr/bin/env python3
"""
Network conditions publisher for the EdgeLab Network VM (172.22.174.148).

Polls the active tc qdisc rules on the given interface and publishes a
structured JSON record to a Kafka topic every --interval-s seconds.
Also publishes immediately on SIGUSR1 (useful when tc_control.sh is
wrapped to send SIGUSR1 to this process after every rule change).

Designed to run as a plain Python script directly on the Network VM:

    Setup:
        cd /home/mae/network_load
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

    Run:
        python3 -u network_conditions_publisher.py \\
            --kafka-bootstrap-servers 172.22.174.149:9092 \\
            --topic edgelab.network.metrics \\
            --interface ens18 \\
            --interval-s 1

    Environment variables (all overridden by CLI args):
        KAFKA_BROKERS       default: 172.22.174.149:9092
        KAFKA_NET_TOPIC     default: edgelab.network.metrics
        NETWORK_INTERFACE   default: ens18
        POLL_INTERVAL_SEC   default: 1
        LOG_LEVEL           default: INFO

Published JSON structure:
    {
        "timestamp":          float   Unix timestamp
        "source":             str     "network-vm"
        "interface":          str     e.g. "ens18"
        "tc_active":          bool    false when no shaping is active
        "mode":               str     "clear" | "tbf" | "netem" | "netem_tbf"
        "delay_ms":           float   added one-way delay in ms (netem)
        "jitter_ms":          float   delay variation in ms (netem)
        "packet_loss_pct":    float   packet loss percentage (netem)
        "packet_loss_percent":float   same value, alternative key
        "bandwidth":          str     rate string (e.g. "50Mbit") or "unlimited"
        "rate":               str     same as bandwidth
        "tbf_latency":        str|None tbf buffer latency (e.g. "50ms") or null
        "raw":                str     raw output from tc -s qdisc show
    }

Kafka infrastructure:
    Kafka broker:  172.22.174.149:9092
    Kafka UI:      http://172.22.174.149:8080
    Topic:         edgelab.network.metrics
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time

from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Set by the SIGUSR1 handler; cleared after each immediate publish.
_sigusr1_event = threading.Event()


def _sigusr1_handler(signum, frame) -> None:
    """Trigger an immediate publish on the next loop tick."""
    _sigusr1_event.set()


# ---------------------------------------------------------------------------
# tc output parsing
# ---------------------------------------------------------------------------

# Regex fragment: a tc time token like 100us, 5ms, 1.5ms, 1s
_TIME_TOKEN = r"\d+(?:\.\d+)?(?:us|ms|s)"

# Regex fragment: a tc rate token like 50Mbit, 1Gbit, 100Kbit
_RATE_TOKEN = r"\d+(?:\.\d+)?(?:Kbit|Mbit|Gbit|bit)"


def _time_to_ms(token: str) -> float:
    """Convert a tc time token (e.g. '100us', '5ms', '1s') to milliseconds."""
    token = token.strip().lower()
    if token.endswith("us"):
        return float(token[:-2]) / 1000.0
    if token.endswith("ms"):
        return float(token[:-2])
    if token.endswith("s"):
        return float(token[:-1]) * 1000.0
    # Fallback: assume ms
    return float(token)


def parse_tc_rules(interface: str) -> dict:
    """
    Run 'tc -s qdisc show dev <interface>' and parse the active rules.

    Returns a dict with fields:
        delay_ms, jitter_ms, packet_loss_pct, packet_loss_percent,
        bandwidth, rate, tbf_latency, tc_active, mode, interface, raw.

    Three recognised configurations:
        clear      - no active netem/tbf rules (just the default qdisc)
        tbf        - bandwidth shaping only
        netem_tbf  - delay+jitter (netem) with bandwidth shaping (tbf child)
        netem      - delay+jitter only (no tbf; rare but handled)

    All values default safely to 0.0 / "unlimited" / false on parse error.
    """
    result: dict = {
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "packet_loss_pct": 0.0,
        "packet_loss_percent": 0.0,
        "bandwidth": "unlimited",
        "rate": "unlimited",
        "tbf_latency": None,
        "tc_active": False,
        "mode": "clear",
        "interface": interface,
        "raw": "",
    }

    try:
        proc = subprocess.run(
            ["tc", "-s", "qdisc", "show", "dev", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = proc.stdout
        if proc.returncode != 0:
            logger.warning("tc exited %d: %s", proc.returncode, proc.stderr.strip())
    except FileNotFoundError:
        logger.error("'tc' command not found; install iproute2")
        return result
    except subprocess.TimeoutExpired:
        logger.error("tc qdisc query timed out")
        return result
    except Exception as exc:
        logger.error("tc qdisc error: %s", exc)
        return result

    result["raw"] = output.strip()

    # Only look at qdisc config lines (lines starting with "qdisc "); skip
    # statistics lines like "Sent X bytes", "backlog ...", etc.
    config_lines = [
        line for line in output.splitlines()
        if line.strip().startswith("qdisc ")
    ]

    netem_line = next((l for l in config_lines if "netem" in l), "")
    tbf_line = next((l for l in config_lines if "tbf" in l), "")

    if not netem_line and not tbf_line:
        # Only the default qdisc (fq_codel, pfifo_fast, etc.) is active.
        return result

    result["tc_active"] = True

    if netem_line and tbf_line:
        result["mode"] = "netem_tbf"
    elif tbf_line:
        result["mode"] = "tbf"
    else:
        result["mode"] = "netem"

    # --- Parse netem delay / jitter / loss ---
    if netem_line:
        dm = re.search(
            rf"delay\s+({_TIME_TOKEN})(?:\s+({_TIME_TOKEN}))?",
            netem_line,
        )
        if dm:
            result["delay_ms"] = round(_time_to_ms(dm.group(1)), 4)
            if dm.group(2):
                result["jitter_ms"] = round(_time_to_ms(dm.group(2)), 4)

        lm = re.search(r"loss\s+(\d+(?:\.\d+)?)%", netem_line)
        if lm:
            loss = float(lm.group(1))
            result["packet_loss_pct"] = loss
            result["packet_loss_percent"] = loss

    # --- Parse tbf rate / latency ---
    if tbf_line:
        rm = re.search(
            rf"rate\s+({_RATE_TOKEN})",
            tbf_line,
            re.IGNORECASE,
        )
        if rm:
            result["bandwidth"] = rm.group(1)
            result["rate"] = rm.group(1)

        # tc shows the tbf buffer latency as "lat Xms"
        lm = re.search(r"\blat\b\s+(\S+)", tbf_line)
        if lm:
            result["tbf_latency"] = lm.group(1)

    return result


# ---------------------------------------------------------------------------
# CLI and main loop
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish Network VM tc qdisc state to Kafka.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--kafka-bootstrap-servers",
        default=os.environ.get("KAFKA_BROKERS", "172.22.174.149:9092"),
        metavar="BROKERS",
        help="Kafka bootstrap servers (host:port)",
    )
    p.add_argument(
        "--topic",
        default=os.environ.get("KAFKA_NET_TOPIC", "edgelab.network.metrics"),
        help="Kafka topic to publish network metrics to",
    )
    p.add_argument(
        "--interface",
        default=os.environ.get("NETWORK_INTERFACE", "ens18"),
        help="Network interface to inspect with tc",
    )
    p.add_argument(
        "--interval-s",
        type=float,
        default=float(os.environ.get("POLL_INTERVAL_SEC", "1")),
        metavar="SECONDS",
        help="Publish interval in seconds",
    )
    return p.parse_args()


def main() -> None:
    """
    Register SIGUSR1 handler, start Kafka producer, and poll tc rules,
    publishing on the configured interval or immediately on SIGUSR1.
    """
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    args = _parse_args()
    kafka_brokers = args.kafka_bootstrap_servers
    net_topic = args.topic
    interface = args.interface
    poll_interval = args.interval_s

    producer = Producer(
        {
            "bootstrap.servers": kafka_brokers,
            "client.id": "network-conditions-publisher",
        }
    )

    logger.info(
        "Network conditions publisher started: brokers=%s topic=%s interface=%s interval=%.1fs",
        kafka_brokers,
        net_topic,
        interface,
        poll_interval,
    )

    def _delivery_report(err, msg) -> None:
        if err:
            logger.error("Kafka delivery failed: %s", err)

    last_publish = 0.0

    while True:
        now = time.time()
        should_publish = (now - last_publish >= poll_interval) or _sigusr1_event.is_set()

        if should_publish:
            metrics = parse_tc_rules(interface)
            metrics["timestamp"] = time.time()
            metrics["source"] = "network-vm"
            _sigusr1_event.clear()
            last_publish = time.time()

            try:
                producer.produce(
                    net_topic,
                    key="net",
                    value=json.dumps(metrics).encode("utf-8"),
                    callback=_delivery_report,
                )
                producer.poll(0)
                logger.info(
                    "Published: mode=%s tc_active=%s delay_ms=%.3f bandwidth=%s",
                    metrics["mode"],
                    metrics["tc_active"],
                    metrics["delay_ms"],
                    metrics["bandwidth"],
                )
            except Exception as exc:
                logger.error("Kafka produce error: %s", exc)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
