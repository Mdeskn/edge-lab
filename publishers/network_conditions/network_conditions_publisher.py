"""
Runs on the Network VM. Parses tc qdisc rules and publishes network conditions
to Kafka. Publishes every POLL_INTERVAL_SEC (default 2). Also publishes
immediately on SIGUSR1 (sent by tc_apply.sh / tc_clear.sh after each change).

Environment variables required:
    KAFKA_BROKERS, KAFKA_NET_TOPIC, NETWORK_INTERFACE (default: eth0),
    POLL_INTERVAL_SEC (default: 2), LOG_LEVEL

Default topic: /edgelab/network/metrics
"""
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

_sigusr1_event = threading.Event()


def _sigusr1_handler(signum, frame) -> None:
    """Set the event flag so the main loop publishes immediately."""
    _sigusr1_event.set()


def parse_tc_rules(interface: str) -> dict:
    """
    Read netem qdisc rules from the given interface via 'tc qdisc show'.

    Parses delay, jitter, and packet loss from the netem rule if present.
    Returns a dict:
        delay_ms (float), jitter_ms (float), packet_loss_pct (float), interface (str)
    Missing fields default to 0.0. Returns the same structure (all zeros)
    when no netem rule is found or on error.
    """
    result = {
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "packet_loss_pct": 0.0,
        "interface": interface,
    }

    try:
        proc = subprocess.run(
            ["tc", "qdisc", "show", "dev", interface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = proc.stdout
    except Exception as exc:
        logger.error("tc qdisc error: %s", exc)
        return result

    if "netem" not in output:
        return result

    delay_match = re.search(r"delay\s+(\d+(?:\.\d+)?)ms(?:\s+(\d+(?:\.\d+)?)ms)?", output)
    if delay_match:
        result["delay_ms"] = float(delay_match.group(1))
        if delay_match.group(2):
            result["jitter_ms"] = float(delay_match.group(2))

    loss_match = re.search(r"loss\s+(\d+(?:\.\d+)?)%", output)
    if loss_match:
        result["packet_loss_pct"] = float(loss_match.group(1))

    return result


def main() -> None:
    """
    Register SIGUSR1 handler, initialise Kafka producer, and poll tc rules
    in a tight loop (sleep 0.1s per tick), publishing on elapsed interval
    or SIGUSR1 signal.
    """
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    kafka_brokers = os.environ["KAFKA_BROKERS"]
    net_topic = os.environ.get("KAFKA_NET_TOPIC", "/edgelab/network/metrics")
    interface = os.environ.get("NETWORK_INTERFACE", "eth0")
    poll_interval = float(os.environ.get("POLL_INTERVAL_SEC", "2"))

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

    def _delivery_report(err, msg):
        if err:
            logger.error("Kafka delivery failed: %s", err)

    last_publish = 0.0

    while True:
        now = time.time()
        should_publish = (now - last_publish >= poll_interval) or _sigusr1_event.is_set()

        if should_publish:
            metrics = parse_tc_rules(interface)
            metrics["timestamp"] = time.time()
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
                logger.debug("Published network conditions: %s", metrics)
            except Exception as exc:
                logger.error("Kafka produce error: %s", exc)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
