#!/bin/bash
# Stress Triton with concurrent requests using perf_analyzer.
# Usage: ./gpu_stressor.sh [model_name] [concurrency] [duration_seconds]

MODEL=${1:-yolov10n}
CONCURRENCY=${2:-50}
DURATION_SEC=${3:-30}
TRITON_URL=${TRITON_URL:-localhost:8000}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting GPU stressor: model=$MODEL concurrency=$CONCURRENCY duration=${DURATION_SEC}s"

perf_analyzer \
    -m "$MODEL" \
    -u "$TRITON_URL" \
    --concurrency-range "$CONCURRENCY" \
    --measurement-interval $((DURATION_SEC * 1000)) \
    --shape images:1,3,640,640

EXIT_CODE=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU stressor finished (exit code: $EXIT_CODE)"
exit $EXIT_CODE
