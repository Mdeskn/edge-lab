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
    Phase-and-metric hybrid placement strategy.

    The phase signal reacts early because SeQaM announces a phase before the
    measured GPU/network metrics fully move. The metric checks catch unexpected
    behavior and recover when conditions are better than expected.
    """

    BAD_PHASES = {"gpu_load", "jitter_light", "bandwidth_50", "mixed"}

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

        avg_latency = self.avg_latency or 0

        desired = "remote"

        # Predictive phase signal from SeQaM.
        if phase in self.BAD_PHASES:
            desired = "local"

        # Reactive server-side signals.
        if yolo_queue > 50 or gpu_util > 85 or total_pending > 10:
            desired = "local"

        # Reactive network-side signals.
        if net_delay > 30 or net_jitter > 10 or packet_loss >= 1:
            desired = "local"
        if bandwidth not in ("unlimited", "none", "") and "gbit" not in bandwidth:
            desired = "local"

        # Lagging end-to-end signal. Tune this after running benchmark_inference.py.
        if avg_latency > 250:
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
