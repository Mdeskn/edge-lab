"""
Detects the one-off latency spike that appears right as the gpu_load phase
starts (Triton absorbing the new load) and flags it for exclusion from the
score and dashboard charts.

The flag never touches the latency value itself and is not applied to the
Kafka stream the SP-Agent reads: placement decisions must still see the real
spike so students can react to it. Only score aggregation and chart history
are expected to consult the flag.
"""
import time

TARGET_PHASE = "gpu_load"


class WarmupSpikeFilter:
    """
    Flags at most one latency sample per gpu_load phase entry as a warm-up
    spike: a sample seen within a short settle window right after the phase
    starts that is a large outlier against the pre-transition baseline.
    """

    def __init__(
        self,
        enabled: bool,
        settle_sec: float,
        multiplier: float,
        floor_ms: float,
        ema_alpha: float = 0.2,
    ):
        self._enabled = enabled
        self._settle_sec = max(0.0, settle_sec)
        self._multiplier = max(1.0, multiplier)
        self._floor_ms = max(0.0, floor_ms)
        self._ema_alpha = ema_alpha

        self._last_phase: str | None = None
        self._window_deadline: float | None = None
        self._excluded_in_window = False
        self._baseline_ms: float | None = None

    def check(self, phase: str, latency_ms: float, now: float | None = None) -> bool:
        """
        Update internal state for one scored frame and return True if it
        should be excluded from the score and dashboard charts.
        """
        if not self._enabled:
            return False

        now = time.monotonic() if now is None else now

        if phase != self._last_phase:
            if phase == TARGET_PHASE:
                self._window_deadline = now + self._settle_sec
                self._excluded_in_window = False
            self._last_phase = phase

        in_window = (
            phase == TARGET_PHASE
            and self._window_deadline is not None
            and now < self._window_deadline
        )

        if in_window:
            if not self._excluded_in_window and self._baseline_ms is not None:
                threshold = max(self._multiplier * self._baseline_ms, self._floor_ms)
                if latency_ms > threshold:
                    self._excluded_in_window = True
                    return True
            return False

        # Steady state: fold this sample into the rolling baseline so the
        # next gpu_load entry has a fresh pre-transition reference point.
        self._baseline_ms = (
            latency_ms
            if self._baseline_ms is None
            else (1 - self._ema_alpha) * self._baseline_ms + self._ema_alpha * latency_ms
        )
        return False
