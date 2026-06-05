"""
Runs on the Network VM (172.22.174.148). Parses tc qdisc rules on interface
ens18 and publishes network conditions to Kafka every POLL_INTERVAL_SEC seconds.
Also publishes immediately on SIGUSR1 (sent by tc_control.sh after each change).

Designed to run as a plain Python script on the Network VM without Docker:
    python network_conditions_publisher.py

Environment variables:
    KAFKA_BROKERS       required  e.g. 172.22.174.149:9092
    KAFKA_NET_TOPIC     optional  default: edgelab.network.metrics
    NETWORK_INTERFACE   optional  default: ens18
    POLL_INTERVAL_SEC   optional  default: 2
    LOG_LEVEL           optional  default: INFO
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
    Read netem and tbf qdisc rules from the given interface via 'tc qdisc show'.

    Parses delay, jitter, and packet loss from the netem rule if present.
    Returns a dict:
        delay_ms (float), jitter_ms (float),
        packet_loss_pct (float), packet_loss_percent (float),
        bandwidth (str), interface (str), raw (str)

    packet_loss_pct and packet_loss_percent carry the same value; both keys
    are included for compatibility with different consumers.

    Missing fields default to 0.0 / "unknown". Returns the same structure
    (all zeros) when no netem rule is found or on error.

    TODO: parse tbf rate token (e.g. "rate 10Mbit") for bandwidth field.
    """
    result = {
        "delay_ms": 0.0,
        "jitter_ms": 0.0,
        "packet_loss_pct": 0.0,
        "packet_loss_percent": 0.0,
        "bandwidth": "unknown",
        "interface": interface,
        "raw": "",
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

    result["raw"] = output.strip()

    if "netem" not in output:
        return result

    delay_match = re.search(r"delay\s+(\d+(?:\.\d+)?)ms(?:\s+(\d+(?:\.\d+)?)ms)?", output)
    if delay_match:
        result["delay_ms"] = float(delay_match.group(1))
        if delay_match.group(2):
            result["jitter_ms"] = float(delay_match.group(2))

    loss_match = re.search(r"loss\s+(\d+(?:\.\d+)?)%", output)
    if loss_match:
        loss = float(loss_match.group(1))
        result["packet_loss_pct"] = loss
        result["packet_loss_percent"] = loss

    return result


def main() -> None:
    """
    Register SIGUSR1 handler, initialise Kafka producer, and poll tc rules
    in a tight loop (sleep 0.1s per tick), publishing on elapsed interval
    or SIGUSR1 signal.
    """
    signal.signal(signal.SIGUSR1, _sigusr1_handler)

    kafka_brokers = os.environ["KAFKA_BROKERS"]
    net_topic = os.environ.get("KAFKA_NET_TOPIC", "edgelab.network.metrics")
    interface = os.environ.get("NETWORK_INTERFACE", "ens18")
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
                logger.debug("Published network conditions: %s", metrics)
            except Exception as exc:
                logger.error("Kafka produce error: %s", exc)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
