#!/usr/bin/env python3
"""
Random GPU load generator for EdgeLab Task 4.

Runs on the load VM (emulate@172.22.229.235). Generates bursts of random-concurrency
GPU load against resnet50_full on Triton, separated by random idle gaps.
The SP-Agent cannot predict load from phase name alone — it must react to
observed Triton queue and utilisation metrics.

Usage (on the load VM):
    python3 random_load.py
    python3 random_load.py --duration 120 --max-concurrency 8
    python3 random_load.py --duration 0          # run until Ctrl+C

Environment (all have defaults):
    TRITON_URL          default: 172.22.174.145:8001
    MODEL               default: resnet50_full
    SDK_IMAGE           default: nvcr.io/nvidia/tritonserver:26.01-py3-sdk
    CONTAINER_NAME      default: edgelab_random_load
"""
import argparse
import logging
import os
import random
import signal
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

TRITON_URL = os.environ.get("TRITON_URL", "172.22.174.145:8001")
MODEL = os.environ.get("MODEL", "resnet50_full")
IMAGE = os.environ.get("SDK_IMAGE", "nvcr.io/nvidia/tritonserver:26.01-py3-sdk")
NAME = os.environ.get("CONTAINER_NAME", "edgelab_random_load")


def _container_running() -> bool:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    return NAME in result.stdout.splitlines()


def _start_load(concurrency: int) -> None:
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    subprocess.Popen(
        [
            "docker", "run", "--rm", "--name", NAME, "--net=host", IMAGE,
            "perf_analyzer",
            "-m", MODEL,
            "-i", "grpc",
            "-u", TRITON_URL,
            "--input-data", "random",
            "--concurrency-range", str(concurrency),
            "--measurement-interval", "999999",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("Load ON  concurrency=%d  model=%s  url=%s", concurrency, MODEL, TRITON_URL)


def _stop_load() -> None:
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    logger.info("Load OFF")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random GPU load generator for EdgeLab Task 4.")
    p.add_argument("--duration", type=float, default=120.0,
                   help="Total run time in seconds. 0 = run until Ctrl+C.")
    p.add_argument("--min-concurrency", type=int, default=1,
                   help="Minimum perf_analyzer concurrency.")
    p.add_argument("--max-concurrency", type=int, default=8,
                   help="Maximum perf_analyzer concurrency (keep ≤10 to avoid huge delays).")
    p.add_argument("--min-load-sec", type=float, default=5.0,
                   help="Minimum duration of each load burst (seconds).")
    p.add_argument("--max-load-sec", type=float, default=20.0,
                   help="Maximum duration of each load burst (seconds).")
    p.add_argument("--min-idle-sec", type=float, default=3.0,
                   help="Minimum idle gap between bursts (seconds).")
    p.add_argument("--max-idle-sec", type=float, default=15.0,
                   help="Maximum idle gap between bursts (seconds).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    def _cleanup(signum=None, frame=None) -> None:
        _stop_load()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    logger.info(
        "Random load started: duration=%.0fs concurrency=[%d,%d] "
        "load=[%.0f,%.0f]s idle=[%.0f,%.0f]s",
        args.duration,
        args.min_concurrency, args.max_concurrency,
        args.min_load_sec, args.max_load_sec,
        args.min_idle_sec, args.max_idle_sec,
    )

    start_time = time.monotonic()

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                break

            concurrency = random.randint(args.min_concurrency, args.max_concurrency)
            load_sec = random.uniform(args.min_load_sec, args.max_load_sec)
            idle_sec = random.uniform(args.min_idle_sec, args.max_idle_sec)

            if args.duration > 0:
                remaining = args.duration - (time.monotonic() - start_time)
                if remaining <= 0:
                    break
                load_sec = min(load_sec, remaining)

            _start_load(concurrency)
            time.sleep(load_sec)
            _stop_load()

            if args.duration > 0:
                remaining = args.duration - (time.monotonic() - start_time)
                if remaining <= 0:
                    break
                idle_sec = min(idle_sec, remaining)

            logger.info("Idle for %.1fs", idle_sec)
            time.sleep(idle_sec)

    finally:
        _stop_load()

    logger.info("Done. Total time: %.1fs", time.monotonic() - start_time)


if __name__ == "__main__":
    main()
