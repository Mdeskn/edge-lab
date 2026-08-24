"""Pipeline threads: FrameReader, Dispatcher, Scorer, LatencyProbe."""
# Modules are imported by their full path (e.g. `from threads.scorer import
# Scorer`), so nothing is re-exported here. Eager re-exports made importing
# any one module pull in onnxruntime, tritonclient, and OpenCV.
