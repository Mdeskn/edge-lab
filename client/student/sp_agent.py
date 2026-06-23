"""
YOUR SP-AGENT: Edit ONLY this file.

This is your starting template. Implement the decide() method to control
where inference runs at each moment. Return "local" to run YOLO on the Pi
CPU. Return "remote" to send the frame to the GPU server.

The decide() method is called every 500 ms. Read the docstring in
sp_agent_base.py for the full list of available metrics and helpers.

Tips:
- Read sp_agent_example.py for a reference strategy you can learn from.
- Do NOT submit sp_agent_example.py as your own work.
- Avoid relying on self.experiment_phase as your only signal. Real edge
  systems do not know the experiment phase ahead of time.
- Use .get() on metric dicts: they may be empty until the first Kafka
  message arrives.
"""
from config import Config
from shared_state import SharedState
from student.sp_agent_base import SPAgentBase


class SPAgent(SPAgentBase):
    """
    Your Service Placement Agent.

    Override decide() to return "local" or "remote".

    Available signals (full list in sp_agent_base.py):
        self.experiment_phase                          # current phase string
        self.gpu_metrics.get("gpu_util_pct", 0)        # 0-100
        self.gpu_metrics.get("yolo_queue_ms", 0)       # ms in Triton queue
        self.gpu_metrics.get("yolo_infer_ms", 0)       # GPU inference time
        self.gpu_metrics.get("total_pending", 0)       # pending requests
        self.net_metrics.get("delay_ms", 0)            # added one-way delay
        self.net_metrics.get("jitter_ms", 0)           # delay variation
        self.net_metrics.get("packet_loss_pct", 0)     # 0-100
        self.avg_latency                               # rolling mean of recent
        self.avg_remote_latency                        # rolling active-remote mean
        self.recent_latencies                          # deque of last samples
        self.last_remote_probe_latency                 # inactive remote probe
        self.last_remote_probe_status                  # ok, failed, unavailable
        self.last_remote_probe_age_sec                 # seconds since that probe
        self.last_remote_probe_received_at             # Unix time of that probe
        self.current_mode                              # "local" or "remote"
    """

    def __init__(self, config: Config, shared_state: SharedState):
        """Initialise the agent. Add your own state below if needed."""
        super().__init__(config, shared_state)
        # Add your own internal state here if your strategy needs it.
        # Example:
        # self.consecutive_high_latency = 0

    def decide(self) -> str:
        """
        Return "local" or "remote".

        TODO: replace this stub with your own placement strategy.
        The default below always runs locally, which is a safe but
        suboptimal baseline.
        """
        return "local"
