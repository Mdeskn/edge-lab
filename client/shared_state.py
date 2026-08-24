"""
Thread-safe shared state for all threads in the pipeline.
Every field is accessed through getter/setter methods protected by a single lock.
"""
import threading
from enum import StrEnum


class CollectionState(StrEnum):
    """Lifecycle of the per-cycle data collection."""
    DISCONNECTED = "disconnected"
    ARMED = "armed"
    COLLECTING = "collecting"
    COMPLETE = "complete"


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

        # Held for the duration of a local ONNX inference. The latency probe
        # takes it non-blockingly so its own CPU-bound local inference can
        # never overlap a scored frame's, which would inflate the very latency
        # the student is graded on.
        self.local_inference_lock = threading.Lock()

        self._current_gt_x: float | None = None
        self._current_gt_y: float | None = None
        self._current_frame_number: int = 0
        self._current_video_cycle: int = 0
        self._latest_frame_number: int | None = None
        self._latest_frame = None
        self._latest_prediction: dict | None = None

        self._processing_mode: str = self._validate_processing_mode(initial_mode)
        self._manual_processing_lock: bool = False

        self._cumulative_displacement: float = 0.0
        self._frames_processed: int = 0

        self._experiment_phase: str = "unknown"

        # ─── Collection state machine ────────────────────────────────────────
        self._collection_state: CollectionState = CollectionState.DISCONNECTED
        self._cycle_started_at: float | None = None
        self._cycle_completed_at: float | None = None
        self._cycle_phases_seen: set[str] = set()
        self._cycles_completed: int = 0
        self._previous_phase: str | None = None
        self._armed_for_next_cycle: bool = False
        self._final_cumulative_displacement: float | None = None
        self._cycle_duration_sec: float = 75.0

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

    def lock_processing_mode(self, mode: str) -> None:
        """Force a user-selected placement until another manual command changes it."""
        with self._lock:
            self._processing_mode = self._validate_processing_mode(mode)
            self._manual_processing_lock = True

    def unlock_processing_mode(self) -> None:
        """Allow SP-Agent decisions to update placement again."""
        with self._lock:
            self._manual_processing_lock = False

    def is_processing_mode_locked(self) -> bool:
        """Return True when a dashboard/manual placement command owns placement."""
        with self._lock:
            return self._manual_processing_lock

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
        gt_x: float | None,
        gt_y: float | None,
        video_cycle: int = 0,
    ) -> None:
        """Update the current ground truth coordinates for a given frame."""
        with self._lock:
            self._current_frame_number = frame_number
            self._current_gt_x = gt_x
            self._current_gt_y = gt_y
            self._current_video_cycle = video_cycle

    def get_ground_truth(self) -> tuple:
        """
        Return (frame_number, gt_x, gt_y, video_cycle).

        `video_cycle` counts how many times the clip has looped. The Scorer
        compares it against the cycle a result was captured in: the two differ
        only when the video wrapped while that frame was in flight, and the
        current ground truth then refers to the far end of the clip rather
        than to where the tracked object actually moved.
        """
        with self._lock:
            return (
                self._current_frame_number,
                self._current_gt_x,
                self._current_gt_y,
                self._current_video_cycle,
            )

    # ─── Collection state machine API ────────────────────────────────────

    def get_collection_state(self) -> CollectionState:
        with self._lock:
            return self._collection_state

    def is_collecting(self) -> bool:
        """Return True only when scored frames should be counted toward the cycle."""
        with self._lock:
            return self._collection_state == CollectionState.COLLECTING

    def set_cycle_duration_sec(self, duration: float) -> None:
        """Set the expected duration of one cycle, derived from the scenario file."""
        with self._lock:
            self._cycle_duration_sec = float(duration) if duration > 0 else 75.0

    def get_cycle_duration_sec(self) -> float:
        with self._lock:
            return self._cycle_duration_sec

    def get_collection_snapshot(self) -> dict:
        """Return a thread-safe snapshot of all cycle state for the dashboard."""
        with self._lock:
            return {
                "state": self._collection_state.value,
                "cycle_started_at": self._cycle_started_at,
                "cycle_completed_at": self._cycle_completed_at,
                "phases_seen": sorted(self._cycle_phases_seen),
                "cycles_completed": self._cycles_completed,
                "armed_for_next_cycle": self._armed_for_next_cycle,
                "cycle_duration_sec": self._cycle_duration_sec,
                "final_cumulative_displacement": self._final_cumulative_displacement,
            }

    def transition_to_armed(self) -> None:
        """Move to ARMED. Reset per-cycle counters but keep cycles_completed."""
        with self._lock:
            self._collection_state = CollectionState.ARMED
            self._cycle_started_at = None
            self._cycle_completed_at = None
            self._cycle_phases_seen = set()
            self._armed_for_next_cycle = False
            self._final_cumulative_displacement = None
            self._previous_phase = None
            self._reset_cycle_counters()

    def request_start_on_next_cycle(self) -> None:
        """Set the flag that triggers collection on the next baseline boundary."""
        with self._lock:
            if self._collection_state == CollectionState.ARMED:
                self._armed_for_next_cycle = True

    def transition_to_collecting(self, started_at: float) -> bool:
        """
        Begin collection. Returns True if the transition happened, False if the
        state machine was not in ARMED+armed_for_next_cycle.
        """
        with self._lock:
            if self._collection_state != CollectionState.ARMED:
                return False
            if not self._armed_for_next_cycle:
                return False
            self._collection_state = CollectionState.COLLECTING
            self._cycle_started_at = started_at
            self._cycle_phases_seen = {"baseline"}
            self._armed_for_next_cycle = False
            return True

    def add_phase_seen(self, phase: str) -> None:
        with self._lock:
            if self._collection_state == CollectionState.COLLECTING:
                self._cycle_phases_seen.add(phase)

    def transition_to_complete(self, completed_at: float, final_score: float) -> None:
        with self._lock:
            self._collection_state = CollectionState.COMPLETE
            self._cycle_completed_at = completed_at
            self._cycles_completed += 1
            self._final_cumulative_displacement = final_score

    def get_previous_phase(self) -> str | None:
        with self._lock:
            return self._previous_phase

    def set_previous_phase(self, phase: str) -> None:
        with self._lock:
            self._previous_phase = phase

    def _reset_cycle_counters(self) -> None:
        """Internal: clear per-cycle aggregates. Caller must hold the lock."""
        self._cumulative_displacement = 0.0
        self._frames_processed = 0
        self._phase_scores = {}

    # --- Latest raw frame for optional latency probes ---

    def update_latest_frame(self, frame_number: int, frame) -> None:
        """Store the newest raw video frame for low-rate latency probes."""
        with self._lock:
            self._latest_frame_number = frame_number
            self._latest_frame = frame.copy()

    def get_latest_frame(self) -> tuple[int | None, object | None]:
        """Return (frame_number, frame_copy) for the newest raw frame."""
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame_number, self._latest_frame.copy()

    # --- Latest completed prediction for the real-time dashboard preview ---

    def update_latest_prediction(
        self,
        frame_number: int,
        video_cycle: int,
        predicted_x: float,
        predicted_y: float,
        processing_mode: str,
        latency_ms: float,
    ) -> None:
        """Store the newest completed prediction without changing scoring state."""
        with self._lock:
            self._latest_prediction = {
                "frame_number": frame_number,
                "video_cycle": video_cycle,
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
                "processing_mode": processing_mode,
                "latency_ms": latency_ms,
            }

    def get_latest_prediction(self) -> dict | None:
        """Return a copy of the newest completed prediction."""
        with self._lock:
            return (
                dict(self._latest_prediction)
                if self._latest_prediction is not None
                else None
            )

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

    def add_phase_result(
        self,
        phase: str,
        displacement_px: float | None,
        latency_ms: float,
        jitter_ms: float,
        processing_mode: str,
        deadline_miss: bool = False,
    ) -> None:
        """Record scored per-phase latency, jitter, displacement, deadline miss, and placement split."""
        if displacement_px is None:
            return

        with self._lock:
            self._ensure_phase_stats(phase)
            stats = self._phase_scores[phase]
            stats["total_displacement"] += displacement_px
            stats["frames"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["latency_frames"] += 1
            stats["total_jitter_ms"] += jitter_ms
            stats["jitter_frames"] += 1
            if deadline_miss:
                stats["deadline_misses"] += 1
            if processing_mode.startswith("remote"):
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
                "total_jitter_ms": 0.0,
                "jitter_frames": 0,
                "deadline_misses": 0,
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
