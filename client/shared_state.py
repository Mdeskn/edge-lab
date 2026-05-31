"""
Thread-safe shared state for all threads in the pipeline.
Every field is accessed through getter/setter methods protected by a single lock.
"""
import threading
from collections import deque
from typing import Optional


class SharedState:
    """
    Central shared state object accessed by all four threads.
    All public methods are thread-safe via a single reentrant lock.
    """

    def __init__(self, initial_mode: str = "local"):
        """Initialize shared state with default values and a single mutex."""
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self._current_gt_x: float = 0.0
        self._current_gt_y: float = 0.0
        self._current_frame_number: int = 0

        self._processing_mode: str = initial_mode

        self._recent_latencies: deque = deque(maxlen=20)

        self._cumulative_displacement: float = 0.0
        self._frames_processed: int = 0

        self._experiment_phase: str = "unknown"

        self._phase_scores: dict = {}
        # Structure: {"baseline": {"total_displacement": 0.0, "frames": 0}, ...}

    # --- Processing mode ---

    def set_processing_mode(self, mode: str) -> None:
        """Set the current processing mode. Raises ValueError if not 'local' or 'remote'."""
        if mode not in ("local", "remote"):
            raise ValueError(f"Invalid processing mode: {mode!r}. Must be 'local' or 'remote'.")
        with self._lock:
            self._processing_mode = mode

    def get_processing_mode(self) -> str:
        """Return the current processing mode ('local' or 'remote')."""
        with self._lock:
            return self._processing_mode

    # --- Ground truth ---

    def update_ground_truth(self, frame_number: int, gt_x: float, gt_y: float) -> None:
        """Update the current ground truth coordinates for a given frame."""
        with self._lock:
            self._current_frame_number = frame_number
            self._current_gt_x = gt_x
            self._current_gt_y = gt_y

    def get_ground_truth(self) -> tuple:
        """Return (frame_number, gt_x, gt_y) as a tuple."""
        with self._lock:
            return (self._current_frame_number, self._current_gt_x, self._current_gt_y)

    # --- Latency history ---

    def add_latency(self, latency_ms: float) -> None:
        """Append a latency measurement (ms) to the rolling history."""
        with self._lock:
            self._recent_latencies.append(latency_ms)

    def get_recent_latencies(self) -> list:
        """Return a copy of the recent latency list."""
        with self._lock:
            return list(self._recent_latencies)

    def get_average_latency(self) -> Optional[float]:
        """Return mean of recent latencies, or None if no measurements exist."""
        with self._lock:
            if not self._recent_latencies:
                return None
            return sum(self._recent_latencies) / len(self._recent_latencies)

    # --- Scoring ---

    def add_displacement(self, displacement_px: float) -> None:
        """Atomically increment cumulative displacement and frame count."""
        with self._lock:
            self._cumulative_displacement += displacement_px
            self._frames_processed += 1

    def get_score_summary(self) -> dict:
        """Return a dict with cumulative_displacement, frames_processed, and average_displacement."""
        with self._lock:
            avg = (
                self._cumulative_displacement / self._frames_processed
                if self._frames_processed > 0
                else 0.0
            )
            return {
                "cumulative_displacement": self._cumulative_displacement,
                "frames_processed": self._frames_processed,
                "average_displacement": avg,
            }

    # --- Experiment phase ---

    def update_experiment_phase(self, phase: str) -> None:
        """Update the current experiment phase string."""
        with self._lock:
            self._experiment_phase = phase

    def get_experiment_phase(self) -> str:
        """Return the current experiment phase."""
        with self._lock:
            return self._experiment_phase

    # --- Per-phase scoring ---

    def add_phase_displacement(self, phase: str, displacement_px: float) -> None:
        """Record a displacement measurement for a specific experiment phase."""
        with self._lock:
            if phase not in self._phase_scores:
                self._phase_scores[phase] = {"total_displacement": 0.0, "frames": 0}
            self._phase_scores[phase]["total_displacement"] += displacement_px
            self._phase_scores[phase]["frames"] += 1

    def get_phase_summary(self) -> dict:
        """
        Return per-phase scoring data.
        Returns a dict like:
        {
            "baseline": {"total_displacement": 123.4, "frames": 300},
            "gpu_load": {"total_displacement": 456.7, "frames": 300},
            ...
        }
        """
        with self._lock:
            return {
                phase: dict(scores)
                for phase, scores in self._phase_scores.items()
            }

    # --- Shutdown ---

    def request_shutdown(self) -> None:
        """Signal all threads to stop."""
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Return True if a shutdown has been requested."""
        return self._shutdown_event.is_set()
