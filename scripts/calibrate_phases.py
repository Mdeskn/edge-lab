#!/usr/bin/env python3
"""Sweep GPU/network load and record remote inference latency.

This script is for operators before the lab. It forces the remote path by
calling the JPEG gateway directly, then writes a CSV/table that helps choose
the final SeQaM phase parameters.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests


DEFAULT_GPU_CONCURRENCY = "0,25,50,75,100"
DEFAULT_BANDWIDTHS = "unlimited,200mbit,100mbit,50mbit,20mbit"
DEFAULT_JITTER_MS = "0,10,25,50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep remote GPU concurrency, VM3 bandwidth, and VM3 jitter; "
            "record mean remote JPEG inference latency for each combination."
        )
    )
    parser.add_argument("--vm3-ssh", default=os.environ.get("VM3_SSH", "mae@172.22.174.148"))
    parser.add_argument("--lc1-ssh", default=os.environ.get("LC1_SSH", "emulate@172.22.229.235"))
    parser.add_argument(
        "--tc-script",
        default=os.environ.get("TC_SCRIPT", "/home/mae/network_load/tc_control.sh"),
    )
    parser.add_argument(
        "--gpu-load-script",
        default=os.environ.get(
            "GPU_LOAD_SCRIPT",
            "/home/emulate/edgelab-load-client/run_gpu_load.sh",
        ),
    )
    parser.add_argument(
        "--remote-url",
        default=os.environ.get("REMOTE_INFERENCE_URL", "http://172.22.174.148:8100"),
        help="Router-facing JPEG gateway URL.",
    )
    parser.add_argument("--video", default=os.environ.get("VIDEO_PATH", "data/test_video.mp4"))
    parser.add_argument("--output", default="data/phase_calibration.csv")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=8.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--gpu-concurrency", default=DEFAULT_GPU_CONCURRENCY)
    parser.add_argument("--bandwidths", default=DEFAULT_BANDWIDTHS)
    parser.add_argument("--jitter-ms", default=DEFAULT_JITTER_MS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned SSH commands and combinations without changing anything.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def run_ssh(target: str, command: str, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY RUN ssh {target!r} {command!r}")
        return
    subprocess.run(["ssh", target, command], check=True)


def clear_network(args: argparse.Namespace) -> None:
    run_ssh(args.vm3_ssh, f"sudo -n {args.tc_script} clear", args.dry_run)


def apply_network(args: argparse.Namespace, bandwidth: str, jitter_ms: int) -> None:
    clear_network(args)
    bandwidth = bandwidth.lower()
    if bandwidth == "unlimited" and jitter_ms == 0:
        return

    if bandwidth == "unlimited":
        rate = "1gbit"
        burst = "2mbit"
    else:
        rate = bandwidth
        burst = "512kb" if bandwidth in {"20mbit", "50mbit"} else "2mbit"

    if jitter_ms > 0:
        command = f"sudo -n {args.tc_script} netem_tbf 0ms {jitter_ms}ms {rate} {burst} 50ms"
    else:
        command = f"sudo -n {args.tc_script} tbf {rate} {burst} 50ms"
    run_ssh(args.vm3_ssh, command, args.dry_run)


def stop_gpu_load(args: argparse.Namespace) -> None:
    run_ssh(args.lc1_ssh, f"bash {args.gpu_load_script} stop", args.dry_run)


def apply_gpu_load(args: argparse.Namespace, concurrency: int) -> None:
    stop_gpu_load(args)
    if concurrency > 0:
        run_ssh(
            args.lc1_ssh,
            f"bash {args.gpu_load_script} start {concurrency} 0",
            args.dry_run,
        )


def synthetic_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (220, 170), (420, 310), (40, 180, 240), thickness=-1)
    cv2.putText(
        frame,
        "EdgeLab calibration",
        (120, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode calibration frame as JPEG")
    return jpeg.tobytes()


def load_jpeg_samples(path: str, quality: int, max_samples: int = 90) -> list[bytes]:
    video_path = Path(path)
    samples: list[bytes] = []

    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        while len(samples) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            samples.append(encode_jpeg(frame, quality))
        cap.release()

    if not samples:
        print(
            f"WARNING: {path!r} was not readable; using one synthetic frame.",
            file=sys.stderr,
        )
        samples.append(encode_jpeg(synthetic_frame(), quality))

    return samples


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def measure_remote_latency(
    remote_url: str,
    samples: list[bytes],
    seconds: float,
    timeout: float,
) -> dict[str, object]:
    infer_url = f"{remote_url.rstrip('/')}/infer"
    deadline = time.monotonic() + seconds
    latencies: list[float] = []
    errors = 0
    last_error = ""
    index = 0

    while time.monotonic() < deadline:
        body = samples[index % len(samples)]
        index += 1
        started = time.perf_counter()
        try:
            response = requests.post(
                infer_url,
                data=body,
                headers={"Content-Type": "image/jpeg"},
                timeout=timeout,
            )
            response.raise_for_status()
            latencies.append((time.perf_counter() - started) * 1000.0)
        except Exception as exc:
            errors += 1
            last_error = str(exc)
            time.sleep(0.2)

    return {
        "samples": len(latencies),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "errors": errors,
        "last_error": last_error,
    }


def write_rows(path: str, rows: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gpu_concurrency",
        "bandwidth",
        "jitter_ms",
        "seconds",
        "samples",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "errors",
        "last_error",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows: list[dict[str, object]]) -> None:
    headers = ["gpu", "bandwidth", "jitter", "samples", "mean_ms", "p95_ms", "errors"]
    table_rows = [
        [
            str(row["gpu_concurrency"]),
            str(row["bandwidth"]),
            str(row["jitter_ms"]),
            str(row["samples"]),
            f"{float(row['mean_latency_ms']):.1f}",
            f"{float(row['p95_latency_ms']):.1f}",
            str(row["errors"]),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table_rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> int:
    args = parse_args()
    gpu_levels = [int(value) for value in split_csv(args.gpu_concurrency)]
    bandwidths = split_csv(args.bandwidths)
    jitter_values = [int(value) for value in split_csv(args.jitter_ms)]

    if args.dry_run:
        samples = [b"dry-run"]
    else:
        health = requests.get(f"{args.remote_url.rstrip('/')}/health", timeout=args.request_timeout)
        health.raise_for_status()
        samples = load_jpeg_samples(args.video, args.jpeg_quality)

    rows: list[dict[str, object]] = []
    try:
        for gpu_concurrency in gpu_levels:
            for bandwidth in bandwidths:
                for jitter_ms in jitter_values:
                    print(
                        "calibrating "
                        f"gpu={gpu_concurrency} bandwidth={bandwidth} jitter={jitter_ms}ms"
                    )
                    apply_gpu_load(args, gpu_concurrency)
                    apply_network(args, bandwidth, jitter_ms)
                    if args.dry_run:
                        measurement = {
                            "samples": 0,
                            "mean_latency_ms": 0.0,
                            "p50_latency_ms": 0.0,
                            "p95_latency_ms": 0.0,
                            "errors": 0,
                            "last_error": "",
                        }
                    else:
                        time.sleep(2.0)
                        measurement = measure_remote_latency(
                            args.remote_url,
                            samples,
                            args.seconds,
                            args.request_timeout,
                        )
                    row = {
                        "gpu_concurrency": gpu_concurrency,
                        "bandwidth": bandwidth,
                        "jitter_ms": jitter_ms,
                        "seconds": args.seconds,
                        **measurement,
                    }
                    rows.append(row)
                    write_rows(args.output, rows)
    finally:
        try:
            stop_gpu_load(args)
        finally:
            clear_network(args)

    print()
    print_table(rows)
    print()
    print(f"Wrote calibration CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
