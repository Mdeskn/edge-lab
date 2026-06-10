"""
Benchmark inference latency for local Pi CPU and/or remote GPU inference.

Use --mode local to test only the Pi (no Triton needed).
Use --mode remote to test the JPEG remote inference API (preferred) or legacy Triton.
Use --mode both to test both and get a comparison recommendation.

Run this as the first step before the experiment. The reasoning:
if local on the Pi is 500ms and remote on the H100 is 30ms, then under
any realistic GPU load remote will still win, and the SP-Agent decision
becomes trivial. Both numbers must be measured before designing the
student lab around them.

Usage examples:
    python scripts/benchmark_inference.py --mode local \\
        --model /data/yolov10n.onnx --video /data/video.mp4

    python scripts/benchmark_inference.py --mode remote \\
        --model /data/yolov10n.onnx \\
        --remote-inference-url http://172.22.174.148:8100

    python scripts/benchmark_inference.py --mode both \\
        --model /data/yolov10n.onnx --video /data/video.mp4 \\
        --remote-inference-url http://172.22.174.148:8100 --runs 100 --warmup 10

If --video is omitted, random noise frames are used (valid for latency
measurement; detections will all be empty, which is expected).
"""

import argparse
import statistics
import sys
import time

import cv2
import numpy as np
import requests

# ------------------------------------------------------------------ #
# Preprocessing (mirrors Dispatcher._preprocess exactly)              #
# ------------------------------------------------------------------ #

INPUT_SIZE = 640


def preprocess(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


# ------------------------------------------------------------------ #
# Frame source                                                         #
# ------------------------------------------------------------------ #

def load_sample_frames(video_path: str, n: int) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n)
    frames = []

    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, (i * step) % max(1, total))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if not frames:
        print("ERROR: Could not read any frames from video.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(frames)} sample frames from video ({total} total frames).")
    return frames


def make_random_frames(n: int) -> list:
    print(f"No video provided. Using {n} random noise frames (640x480 BGR).")
    return [
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        for _ in range(n)
    ]


# ------------------------------------------------------------------ #
# Local benchmark                                                      #
# ------------------------------------------------------------------ #

def benchmark_local(model_path: str, frames: list, warmup: int, threads: int = 4) -> list:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime")
        sys.exit(1)

    import os
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preprocessed = [preprocess(f) for f in frames]

    print(f"Warming up local inference ({warmup} runs)...", end=" ", flush=True)
    for i in range(warmup):
        session.run([output_name], {input_name: preprocessed[i % len(preprocessed)]})
    print("done.")

    print(f"Benchmarking local inference ({len(frames)} runs)...", end=" ", flush=True)
    latencies = []
    for p in preprocessed:
        t0 = time.perf_counter()
        session.run([output_name], {input_name: p})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    print("done.")

    return latencies


# ------------------------------------------------------------------ #
# Remote benchmark                                                     #
# ------------------------------------------------------------------ #

def benchmark_remote_jpeg(
    remote_inference_url: str,
    frames: list,
    warmup: int,
    timeout: float,
    jpeg_quality: int,
) -> list:
    """Benchmark the VM2 JPEG remote inference API."""
    base_url = remote_inference_url.rstrip("/")
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        print(
            f"ERROR: Cannot connect to remote inference API at {base_url}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Connected to remote inference API at {base_url}.")

    quality = min(max(jpeg_quality, 1), 100)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    print(f"Warming up remote JPEG inference ({warmup} runs)...", end=" ", flush=True)
    for i in range(warmup):
        ok, jpeg = cv2.imencode(".jpg", frames[i % len(frames)], encode_params)
        if not ok:
            print("\nERROR: Could not encode warmup frame as JPEG.", file=sys.stderr)
            sys.exit(1)
        resp = requests.post(
            f"{base_url}/infer",
            data=jpeg.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=timeout,
        )
        resp.raise_for_status()
    print("done.")

    print(f"Benchmarking remote JPEG inference ({len(frames)} runs)...", end=" ", flush=True)
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            print("\nERROR: Could not encode frame as JPEG.", file=sys.stderr)
            sys.exit(1)
        resp = requests.post(
            f"{base_url}/infer",
            data=jpeg.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=timeout,
        )
        resp.raise_for_status()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    print("done.")

    return latencies


def benchmark_remote_triton(
    triton_url: str, model_name: str, frames: list, warmup: int, timeout: float
) -> list:
    """Benchmark the legacy direct Triton HTTP path with preprocessed FP32 tensors."""
    try:
        import tritonclient.http as httpclient
    except ImportError:
        print(
            "ERROR: tritonclient is not installed. "
            "Run: pip install tritonclient[http]"
        )
        sys.exit(1)

    try:
        client = httpclient.InferenceServerClient(url=triton_url, verbose=False)
        if not client.is_server_live():
            print(
                f"ERROR: Triton server at {triton_url} is not live.", file=sys.stderr
            )
            sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Cannot connect to Triton at {triton_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to Triton at {triton_url}.")

    preprocessed = [preprocess(f) for f in frames]

    print(f"Warming up remote inference ({warmup} runs)...", end=" ", flush=True)
    for i in range(warmup):
        inp = httpclient.InferInput("images", [1, 3, 640, 640], "FP32")
        inp.set_data_from_numpy(preprocessed[i % len(preprocessed)])
        out = httpclient.InferRequestedOutput("output0")
        client.infer(model_name, inputs=[inp], outputs=[out], timeout=timeout)
    print("done.")

    print(f"Benchmarking remote inference ({len(frames)} runs)...", end=" ", flush=True)
    latencies = []
    for p in preprocessed:
        inp = httpclient.InferInput("images", [1, 3, 640, 640], "FP32")
        inp.set_data_from_numpy(p)
        out = httpclient.InferRequestedOutput("output0")
        t0 = time.perf_counter()
        client.infer(model_name, inputs=[inp], outputs=[out], timeout=timeout)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    print("done.")

    return latencies


def benchmark_remote(
    args: argparse.Namespace,
    frames: list,
) -> tuple[str, list]:
    """Benchmark the configured remote endpoint and return (label, latencies)."""
    if args.remote_inference_url:
        label = f"Remote inference (JPEG API at {args.remote_inference_url})"
        latencies = benchmark_remote_jpeg(
            args.remote_inference_url,
            frames,
            args.warmup,
            args.timeout,
            args.remote_jpeg_quality,
        )
        return label, latencies

    label = f"Remote inference (legacy Triton at {args.triton_url})"
    latencies = benchmark_remote_triton(
        args.triton_url, args.model_name, frames, args.warmup, args.timeout
    )
    return label, latencies


# ------------------------------------------------------------------ #
# Stats                                                                #
# ------------------------------------------------------------------ #

def compute_stats(latencies: list) -> dict:
    sorted_l = sorted(latencies)
    p95_idx = int(len(sorted_l) * 0.95)
    p99_idx = int(len(sorted_l) * 0.99)
    return {
        "count": len(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p95": sorted_l[min(p95_idx, len(sorted_l) - 1)],
        "p99": sorted_l[min(p99_idx, len(sorted_l) - 1)],
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


def print_stats(label: str, stats: dict) -> None:
    print(f"\n  {label}")
    print(f"    runs   : {stats['count']}")
    print(f"    mean   : {stats['mean']:.1f} ms")
    print(f"    median : {stats['median']:.1f} ms")
    print(f"    min    : {stats['min']:.1f} ms")
    print(f"    max    : {stats['max']:.1f} ms")
    print(f"    p95    : {stats['p95']:.1f} ms")
    print(f"    p99    : {stats['p99']:.1f} ms")
    print(f"    stdev  : {stats['stdev']:.1f} ms")


def print_recommendation(local_stats: dict, remote_stats: dict) -> None:
    local_mean = local_stats["mean"]
    remote_mean = remote_stats["mean"]
    speedup = local_mean / remote_mean if remote_mean > 0 else float("inf")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"  Local  mean latency : {local_mean:.1f} ms")
    print(f"  Remote mean latency : {remote_mean:.1f} ms")
    print(f"  Speedup (local/remote): {speedup:.1f}x")

    if speedup >= 3.0:
        print(
            "\n  Remote is significantly faster. Under light GPU load,\n"
            "  'always remote' is a strong baseline. Consider switching\n"
            "  to local only when GPU utilization is very high or network\n"
            "  delay exceeds ~{:.0f} ms.".format(local_mean - remote_mean)
        )
    elif speedup >= 1.5:
        print(
            "\n  Remote is moderately faster. A smart SP-Agent that switches\n"
            "  to local during GPU load or high network delay should\n"
            "  outperform both 'always local' and 'always remote'."
        )
    elif speedup >= 0.8:
        print(
            "\n  Local and remote are roughly equal. The right choice depends\n"
            "  heavily on current GPU load and network conditions.\n"
            "  A reactive SP-Agent will add the most value here."
        )
    else:
        print(
            "\n  Local is faster than remote under clean conditions.\n"
            "  Check that Triton is configured correctly and the GPU\n"
            "  is not already under load."
        )


# ------------------------------------------------------------------ #
# Local-only interpretation                                            #
# ------------------------------------------------------------------ #

def print_local_interpretation(stats: dict) -> None:
    mean = stats["mean"]

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"  Local mean latency: {mean:.1f} ms")

    if mean < 100:
        print(
            "\n  Local inference is very fast (under 100ms). This may make\n"
            "  the lab too easy: if local is this quick, remote will need\n"
            "  to be only slightly faster to win. Consider slowing the Pi\n"
            "  down by passing --threads 1, or swapping to a heavier model\n"
            "  such as YOLOv10s, so that remote has a meaningful advantage."
        )
    elif mean < 300:
        print(
            "\n  Local inference is moderately slow (100-300ms). Remote will\n"
            "  likely be faster under clean network and GPU conditions.\n"
            "  A well-designed SP-Agent should show a clear improvement over\n"
            "  both 'always local' and 'always remote'."
        )
    elif mean < 600:
        print(
            "\n  Local inference is in the target range for this lab (300-600ms).\n"
            "  Remote inference on the GPU server should be significantly faster\n"
            "  under clean conditions, but network and GPU load will make the\n"
            "  decision non-trivial. Good conditions for the SP-Agent exercise."
        )
    else:
        print(
            "\n  Local inference is very slow (over 600ms). Even a heavily\n"
            "  loaded GPU server should still be faster. Consider whether\n"
            "  the Pi has other processes competing for CPU, or whether\n"
            "  the model is larger than expected."
        )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv10n inference latency on Pi CPU and/or remote GPU."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["local", "remote", "both"],
        help=(
            "What to benchmark. "
            "'local': Pi CPU only, no Triton needed. "
            "'remote': remote GPU only, requires --remote-inference-url or --triton-url. "
            "'both': run both and print a comparison recommendation."
        ),
    )
    parser.add_argument("--model", required=True, help="Path to yolov10n.onnx")
    parser.add_argument(
        "--video",
        default=None,
        help="Path to video file (optional; random noise frames used if omitted)",
    )
    parser.add_argument(
        "--remote-inference-url",
        default=None,
        help=(
            "Preferred JPEG remote inference API URL, "
            "e.g. http://172.22.174.148:8100"
        ),
    )
    parser.add_argument(
        "--remote-jpeg-quality",
        type=int,
        default=80,
        help="JPEG quality for --remote-inference-url requests (default: 80)",
    )
    parser.add_argument(
        "--triton-url",
        default=None,
        help=(
            "Legacy direct Triton HTTP address, e.g. 192.168.1.100:8000. "
            "Used only when --remote-inference-url is omitted."
        ),
    )
    parser.add_argument(
        "--model-name", default="yolov10n", help="Triton model name (default: yolov10n)"
    )
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of timed inference runs (default: 50)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup runs not counted (default: 5)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Triton request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "CPU threads for onnxruntime (default: 4, used only when local benchmarking runs). "
            "Pass --threads 1 to slow the Pi down significantly, useful if YOLOv10n turns out "
            "to be too fast at full speed for the lab exercise."
        ),
    )
    args = parser.parse_args()

    if args.mode in ("remote", "both") and not (args.remote_inference_url or args.triton_url):
        print(
            f"ERROR: --mode {args.mode} requires --remote-inference-url or --triton-url.\n"
            "Prefer the JPEG API (e.g. --remote-inference-url http://172.22.174.148:8100)\n"
            "or use --mode local to benchmark only the Pi CPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("Edge Lab: Inference Latency Benchmark")
    print("=" * 60)
    print(f"  mode       : {args.mode}")
    print(f"  model      : {args.model}")
    print(f"  video      : {args.video or '(random noise)'}")
    if args.mode in ("remote", "both"):
        if args.remote_inference_url:
            print(f"  remote_url : {args.remote_inference_url}")
            print(f"  jpeg_quality: {args.remote_jpeg_quality}")
        else:
            print(f"  triton_url : {args.triton_url}")
    if args.mode in ("local", "both"):
        print(f"  threads    : {args.threads}")
    print(f"  runs       : {args.runs}")
    print(f"  warmup     : {args.warmup}")
    print()

    total_needed = args.runs + args.warmup
    if args.video:
        frames = load_sample_frames(args.video, total_needed)
    else:
        frames = make_random_frames(total_needed)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if args.mode == "local":
        local_latencies = benchmark_local(args.model, frames, args.warmup, args.threads)
        local_stats = compute_stats(local_latencies)
        print_stats("Local inference (Pi CPU / onnxruntime)", local_stats)
        print_local_interpretation(local_stats)

    elif args.mode == "remote":
        remote_label, remote_latencies = benchmark_remote(args, frames)
        remote_stats = compute_stats(remote_latencies)
        print_stats(remote_label, remote_stats)

    else:  # both
        local_latencies = benchmark_local(args.model, frames, args.warmup, args.threads)
        local_stats = compute_stats(local_latencies)
        print()
        remote_label, remote_latencies = benchmark_remote(args, frames)
        remote_stats = compute_stats(remote_latencies)
        print_stats("Local inference (Pi CPU / onnxruntime)", local_stats)
        print_stats(remote_label, remote_stats)
        print_recommendation(local_stats, remote_stats)

    print()


if __name__ == "__main__":
    main()
