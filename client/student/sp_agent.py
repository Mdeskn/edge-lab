"""
YOUR SP-AGENT: Edit ONLY this file.

Implement the decide() method to control where inference runs.
Read the docstring in sp_agent_base.py for the full list of available metrics.
"""
from student.sp_agent_base import SPAgentBase
from config import Config
from shared_state import SharedState


class SPAgent(SPAgentBase):
    """
    Your Service Placement Agent.

    Implement decide() below. Return "local" or "remote".

    Tips:
    - Do NOT use self.experiment_phase to make decisions: an agent that checks
      phase names instead of real metrics will be penalised. Your logic must
      work based on what the sensors actually report.
    - Check self.avg_latency to see how recent end-to-end performance has been.
    - Use .get() for metric dicts: they may be empty until the first Kafka message arrives.

    GPU metrics examples (see sp_agent_base.py for the full list):
        self.gpu_metrics.get("gpu_util_pct", 0)     # GPU utilization 0-100 %
        self.gpu_metrics.get("yolo_queue_ms", 0)    # time requests wait in Triton queue
        self.gpu_metrics.get("yolo_infer_ms", 0)    # GPU inference time for yolov10n
        self.gpu_metrics.get("total_rps", 0)        # total requests/sec across all models

    Network metrics examples:
        self.net_metrics.get("delay_ms", 0)         # added one-way delay in ms
        self.net_metrics.get("jitter_ms", 0)        # delay variation in ms
        self.net_metrics.get("packet_loss_pct", 0)  # packet loss percentage
    """

    def __init__(self, config: Config, shared_state: SharedState):
        """Initialise the agent. Add your own state below if needed."""
        super().__init__(config, shared_state)
        # Add your own state here if needed
        # Example:
        # self.consecutive_high_latency = 0

    def decide(self) -> str:
        """
        Implement your placement strategy here.
        Return "local" or "remote".

        Use the measured metrics below. Do NOT check self.experiment_phase to
        decide: phase names are not available in a real deployment and will not
        earn marks. Your strategy must react to actual observed signal values.
        """
        gpu_util = self.gpu_metrics.get("gpu_util_pct", 0)
        yolo_queue = self.gpu_metrics.get("yolo_queue_ms", 0)
        net_delay = self.net_metrics.get("delay_ms", 0)
        packet_loss = self.net_metrics.get("packet_loss_pct", 0)

        # GPU server is overloaded: local inference avoids the queue.
        if gpu_util > 80 or yolo_queue > 50:
            return "local"

        # Network path is degraded: extra RTT or loss makes remote slower.
        if net_delay > 30 or packet_loss >= 1.0:
            return "local"

        # Conditions look healthy: remote GPU is faster than local CPU.
        return "remote"
