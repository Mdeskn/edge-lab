"""
Watch the Network VM phase file and publish experiment phase changes to Kafka.

Designed to run as a plain Python script directly on the Network VM (172.22.174.148):
    python tc_controller.py

SeQaM (or a manual operator) writes the current phase name to PHASE_FILE.
This script detects changes and publishes them to KAFKA_PHASE_TOPIC.

tc_control.sh on the Network VM applies the actual traffic-shaping rules.
network_conditions_publisher.py reads and publishes the resulting tc state.
This script only publishes the experiment phase name.

Environment variables:
    KAFKA_BROKERS           required  e.g. 172.22.174.149:9092
    KAFKA_PHASE_TOPIC       optional  default: edgelab.phase
    PHASE_FILE              optional  default: /tmp/edgelab_phase
    PHASE_POLL_INTERVAL_SEC optional  default: 0.1
    LOG_LEVEL               optional  default: INFO
"""
import json
import logging
import os
from pathlib import Path
import time

from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PHASES = {
    "baseline": {
        "description": "No load applied",
        "gpu_load": False,
        "network_load": False,
    },
    "gpu_load": {
        "description": "GPU server under load (100 RPS)",
        "gpu_load": True,
        "network_load": False,
    },
    "network_load": {
        "description": "Network degraded (100ms delay, 20ms jitter, 2% loss)",
        "gpu_load": False,
        "network_load": True,
    },
    "combined": {
        "description": "Both GPU and network load active",
        "gpu_load": True,
        "network_load": True,
    },
}


def read_phase(path: Path) -> str | None:
    """Return a valid phase name from the watched file, if one is available."""
    try:
        phase = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Could not read phase file %s: %s", path, exc)
        return None

    if not phase:
        return None
    if phase not in PHASES:
        logger.warning("Ignoring unknown phase %r in %s", phase, path)
        return None
    return phase


def main() -> None:
    """Publish each observed phase transition once."""
    kafka_brokers = os.environ["KAFKA_BROKERS"]
    topic = os.environ.get("KAFKA_PHASE_TOPIC", "edgelab.phase")
    phase_file = Path(os.environ.get("PHASE_FILE", "/tmp/edgelab_phase"))
    poll_interval = float(os.environ.get("PHASE_POLL_INTERVAL_SEC", "0.1"))

    producer = Producer(
        {
            "bootstrap.servers": kafka_brokers,
            "client.id": "edge-lab-phase-publisher",
        }
    )
    logger.info(
        "Phase publisher started: brokers=%s topic=%s file=%s interval=%.2fs",
        kafka_brokers,
        topic,
        phase_file,
        poll_interval,
    )

    def delivery_report(err, msg) -> None:
        if err:
            logger.error("Kafka delivery failed: %s", err)

    last_phase: str | None = None

    while True:
        phase = read_phase(phase_file)
        if phase and phase != last_phase:
            payload = {
                "phase": phase,
                **PHASES[phase],
                "timestamp": time.time(),
            }
            try:
                producer.produce(
                    topic,
                    key="phase",
                    value=json.dumps(payload).encode("utf-8"),
                    callback=delivery_report,
                )
                producer.poll(0)
                last_phase = phase
                logger.info("Published experiment phase: %s", phase)
            except Exception as exc:
                logger.error("Kafka produce error: %s", exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
