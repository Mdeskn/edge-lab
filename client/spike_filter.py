"""
Detects the one-off latency spike that appears right after a phase transition
(a backend absorbing newly-applied load) and flags it for exclusion from the
score and dashboard charts.

The flag never touches the latency value itself and is not applied to the
Kafka stream the SP-Agent reads: placement decisions must still see the real
spike so students can react to it. Only score aggregation and chart history
are expected to consult the flag.

The filter arms on *every* phase transition by default. It used to arm only
when entering "gpu_load", which meant an identical transition artifact was
excluded from the score in one phase and counted against students in the
other five. Set WARMUP_SPIKE_PHASES to restrict it to named phases.
"""
import time
from collections.abc import Iterable


class WarmupSpikeFilter:
    """
    Flags at most one latency sample per phase entry as a warm-up spike: a
    sample seen within a short settle window right after the phase starts that
    is a large outlier against the pre-transition baseline.
    """

    def __init__(
        self,
        enabled: bool,
        settle_sec: float,
        multiplier: float,
        floor_ms: float,
        phases: Iterable[str] = (),
        ema_alpha: float = 0.2,
    ):
        self._enabled = enabled
        self._settle_sec = max(0.0, settle_sec)
        self._multiplier = max(1.0, multiplier)
        self._floor_ms = max(0.0, floor_ms)
        self._ema_alpha = ema_alpha
        # Empty means every phase transition arms the filter.
        self._phases = frozenset(phases)

        self._last_phase: str | None = None
        self._window_deadline: float | None = None
        self._excluded_in_window = False
        self._baseline_ms: float | None = None
        self._excluded_count = 0

    @property
    def excluded_count(self) -> int:
        """Total samples this filter has flagged, for end-of-run reporting."""
        return self._excluded_count

    def _arms_on(self, phase: str) -> bool:
        """Return True when entering `phase` should open an exclusion window."""
        return not self._phases or phase in self._phases

    def check(self, phase: str, latency_ms: float, now: float | None = None) -> bool:
        """
        Update internal state for one scored frame and return True if it
        should be excluded from the score and dashboard charts.
        """
        if not self._enabled:
            return False

        now = time.monotonic() if now is None else now

        if phase != self._last_phase:
            # The very first sample has no pre-transition baseline to compare
            # against, so startup must not open a window it cannot judge.
            if self._last_phase is not None and self._arms_on(phase):
                self._window_deadline = now + self._settle_sec
                self._excluded_in_window = False
            else:
                # Moving into a phase the filter does not cover closes any
                # window still open, so one phase's window can never leak into
                # the next phase's samples.
                self._window_deadline = None
            self._last_phase = phase

        in_window = (
            self._window_deadline is not None
            and now < self._window_deadline
        )

        if in_window:
            if not self._excluded_in_window and self._baseline_ms is not None:
                threshold = max(self._multiplier * self._baseline_ms, self._floor_ms)
                if latency_ms > threshold:
                    self._excluded_in_window = True
                    self._excluded_count += 1
                    return True
            return False

        # Steady state: fold this sample into the rolling baseline so the
        # next phase entry has a fresh pre-transition reference point.
        self._baseline_ms = (
            latency_ms
            if self._baseline_ms is None
            else (1 - self._ema_alpha) * self._baseline_ms + self._ema_alpha * latency_ms
        )
        return False
