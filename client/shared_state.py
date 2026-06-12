"""
Thread-safe shared state for all threads in the pipeline.
Every field is accessed through getter/setter methods protected by a single lock.
"""
import threading
from collections import deque
from typing import Optional


REQUESTED_PROCESSING_MODES = ("local", "remote")


class SharedState:
    """
    Central shared state object accessed by all four threads.
    All public methods are thread-safe via a single reentrant lock.
    """

    def __init__(self, initial_mode: str = "local"):
        """Initialize shared state with default values and a single mutex."""
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self._current_gt_x: Optional[float] = None
        self._current_gt_y: Optional[float] = None
        self._current_frame_number: int = 0

        self._processing_mode: str = self._validate_processing_mode(initial_mode)

        self._recent_latencies: deque = deque(maxlen=20)

        self._cumulative_displacement: float = 0.0
        self._frames_processed: int = 0

        self._experiment_phase: str = "unknown"

        self._phase_scores: dict = {}
        # Structure:
        # {
        #   "baseline": {
        #       "total_displacement": 0.0,
        #       "frames": 0,
        #       "total_latency_ms": 0.0,
        #       "latency_frames": 0,
        #       "local_frames": 0,
        #       "remote_frames": 0,
        #       "local_fallback_frames": 0,
        #   },
        #   ...
        # }

    # --- Processing mode ---

    def set_processing_mode(self, mode: str) -> None:
        """
        Set the requested inference placement.

        Only SP-Agent choices belong in shared state. Dispatcher result labels
        such as "local_fallback" travel with scored frames instead.
        """
        with self._lock:
            self._processing_mode = self._validate_processing_mode(mode)

    def get_processing_mode(self) -> str:
        """Return the requested inference placement ('local' or 'remote')."""
        with self._lock:
            return self._processing_mode

    @staticmethod
    def _validate_processing_mode(mode: str) -> str:
        """Return a valid requested placement or raise ValueError."""
        if mode not in REQUESTED_PROCESSING_MODES:
            raise ValueError(
                f"Invalid requested processing mode: {mode!r}. "
                "Must be 'local' or 'remote'."
            )
        return mode

    # --- Ground truth ---

    def update_ground_truth(
        self,
        frame_number: int,
        gt_x: Optional[float],
        gt_y: Optional[float],
    ) -> None:
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
            self._ensure_phase_stats(phase)
            self._phase_scores[phase]["total_displacement"] += displacement_px
            self._phase_scores[phase]["frames"] += 1

    def add_phase_result(
        self,
        phase: str,
        displacement_px: float | None,
        latency_ms: float,
        processing_mode: str,
    ) -> None:
        """Record scored per-phase latency, displacement, and placement split."""
        if displacement_px is None:
            return

        with self._lock:
            self._ensure_phase_stats(phase)
            stats = self._phase_scores[phase]
            stats["total_displacement"] += displacement_px
            stats["frames"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["latency_frames"] += 1
            if processing_mode == "remote":
                stats["remote_frames"] += 1
            elif processing_mode == "local_fallback":
                stats["local_fallback_frames"] += 1
            else:
                stats["local_frames"] += 1

    def _ensure_phase_stats(self, phase: str) -> None:
        """Initialise per-phase counters when a phase appears for the first time."""
        if phase not in self._phase_scores:
            self._phase_scores[phase] = {
                "total_displacement": 0.0,
                "frames": 0,
                "total_latency_ms": 0.0,
                "latency_frames": 0,
                "local_frames": 0,
                "remote_frames": 0,
                "local_fallback_frames": 0,
            }

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
