"""
Reference SP-Agent strategy for students to study.

Do not submit this file as-is. Copy ideas from it into sp_agent.py and tune the
thresholds using your own benchmark results.
"""
from config import Config
from shared_state import SharedState
from student.sp_agent_base import SPAgentBase


class ExampleSPAgent(SPAgentBase):
    """
    Metric-driven placement strategy with remote-probe recovery.

    A stress phase initially keeps a local agent local until it has a fresh
    remote probe. A healthy probe lets the agent return to remote instead of
    remaining local for the rest of the phase after one startup spike.
    """

    BAD_PHASES = {"gpu_load", "jitter_light", "bandwidth_50", "mixed"}
    REMOTE_LATENCY_LIMIT_MS = 250.0
    REMOTE_PROBE_MAX_AGE_SEC = 20.0

    def __init__(self, config: Config, shared_state: SharedState):
        """Keep a little state so the strategy avoids switching too rapidly."""
        super().__init__(config, shared_state)
        self._last_decision = self.current_mode
        self._pending_decision = self.current_mode
        self._pending_count = 0

    def decide(self) -> str:
        """Return "local" or "remote" based on phase, metrics, and hysteresis."""
        phase = self.experiment_phase
        gpu_util = self.gpu_metrics.get("gpu_util_pct", 0)
        yolo_queue = self.gpu_metrics.get("yolo_queue_ms", 0)
        total_pending = self.gpu_metrics.get("total_pending", 0)

        net_delay = self.net_metrics.get("delay_ms", 0)
        net_jitter = self.net_metrics.get("jitter_ms", 0)
        packet_loss = self.net_metrics.get(
            "packet_loss_pct",
            self.net_metrics.get("packet_loss_percent", 0),
        )
        bandwidth = str(self.net_metrics.get("bandwidth", "unlimited")).lower()

        live_metrics_require_local = (
            yolo_queue > 50
            or gpu_util > 85
            or total_pending > 10
            or net_delay > 30
            or net_jitter > 10
            or packet_loss >= 1
            or (bandwidth not in ("unlimited", "none", "") and "gbit" not in bandwidth)
        )

        desired = "local" if live_metrics_require_local else "remote"

        # Only active remote inference latency can force an immediate move to
        # local. Local samples should not prevent a later remote recovery.
        if (
            self.current_mode == "remote"
            and (self.avg_remote_latency or 0) > self.REMOTE_LATENCY_LIMIT_MS
        ):
            desired = "local"

        if self.current_mode == "local" and not live_metrics_require_local:
            probe_age = self.last_remote_probe_age_sec
            has_fresh_probe = (
                probe_age is not None
                and probe_age <= self.REMOTE_PROBE_MAX_AGE_SEC
            )
            if has_fresh_probe:
                desired = (
                    "remote"
                    if (
                        self.last_remote_probe_status == "ok"
                        and self.last_remote_probe_latency is not None
                        and self.last_remote_probe_latency <= self.REMOTE_LATENCY_LIMIT_MS
                    )
                    else "local"
                )
            elif phase in self.BAD_PHASES:
                # Wait for the first remote probe rather than repeatedly
                # reintroducing remote traffic after a bad-phase spike.
                desired = "local"

        return self._stable_decision(desired)

    def _stable_decision(self, desired: str) -> str:
        """
        Require the same new decision twice before switching.

        This simple hysteresis prevents mode thrashing when a metric hovers near
        a threshold.
        """
        if desired == self._last_decision:
            self._pending_decision = desired
            self._pending_count = 0
            return self._last_decision

        if desired == self._pending_decision:
            self._pending_count += 1
        else:
            self._pending_decision = desired
            self._pending_count = 1

        if self._pending_count >= 2:
            self._last_decision = desired
            self._pending_count = 0

        return self._last_decision
SPAgent = ExampleSPAgent
