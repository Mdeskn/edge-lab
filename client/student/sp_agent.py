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
    - Check self.experiment_phase to know what load is currently running
    - Check self.avg_latency to see how recent performance has been
    - Use .get() for metric dicts: they are empty until the first Kafka message arrives.
      Example: self.gpu_metrics.get("gpu_utilization_pct", 0)
               self.net_metrics.get("delay_ms", 0)
    - You can add your own state in __init__ (e.g. counters, thresholds)
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

        Example starter (always local; replace this with your logic):
        """
        # TODO: Replace with your logic
        return "local"
