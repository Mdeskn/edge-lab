"""
Benchmark inference latency for local Pi CPU and/or remote Triton GPU.

Use --mode local to test only the Pi (no Triton needed).
Use --mode remote to test only Triton (requires --triton-url).
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
        --model /data/yolov10n.onnx --triton-url 192.168.1.100:8000

    python scripts/benchmark_inference.py --mode both \\
        --model /data/yolov10n.onnx --video /data/video.mp4 \\
        --triton-url 192.168.1.100:8000 --runs 100 --warmup 10

If --video is omitted, random noise frames are used (valid for latency
measurement; detections will all be empty, which is expected).
"""

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

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

def benchmark_remote(
    triton_url: str, model_name: str, frames: list, warmup: int, timeout: float
) -> list:
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
        description="Benchmark YOLOv10n inference latency on Pi CPU and/or Triton GPU."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["local", "remote", "both"],
        help=(
            "What to benchmark. "
            "'local': Pi CPU only, no Triton needed. "
            "'remote': Triton only, requires --triton-url. "
            "'both': run both and print a comparison recommendation, requires --triton-url."
        ),
    )
    parser.add_argument("--model", required=True, help="Path to yolov10n.onnx")
    parser.add_argument(
        "--video",
        default=None,
        help="Path to video file (optional; random noise frames used if omitted)",
    )
    parser.add_argument(
        "--triton-url",
        default=None,
        help="Triton server address, e.g. 192.168.1.100:8000 (required for --mode remote and --mode both)",
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

    if args.mode in ("remote", "both") and not args.triton_url:
        print(
            f"ERROR: --mode {args.mode} requires --triton-url.\n"
            "Provide the Triton server address (e.g. --triton-url 192.168.1.100:8000)\n"
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
        remote_latencies = benchmark_remote(
            args.triton_url, args.model_name, frames, args.warmup, args.timeout
        )
        remote_stats = compute_stats(remote_latencies)
        print_stats(f"Remote inference (Triton at {args.triton_url})", remote_stats)

    else:  # both
        local_latencies = benchmark_local(args.model, frames, args.warmup, args.threads)
        local_stats = compute_stats(local_latencies)
        print()
        remote_latencies = benchmark_remote(
            args.triton_url, args.model_name, frames, args.warmup, args.timeout
        )
        remote_stats = compute_stats(remote_latencies)
        print_stats("Local inference (Pi CPU / onnxruntime)", local_stats)
        print_stats(f"Remote inference (Triton at {args.triton_url})", remote_stats)
        print_recommendation(local_stats, remote_stats)

    print()


if __name__ == "__main__":
    main()
