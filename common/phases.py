"""The experiment phase vocabulary, defined once for every component.

The SeQaM scenario on VM1 drives these phases in order and publishes each
transition to KAFKA_PHASE_TOPIC. The client, the dashboard, and the scoring
CSVs must all agree on the names and the ordering, so they all import from
here rather than repeating a literal list.
"""

EXPERIMENT_PHASES: tuple[str, ...] = (
    "cycle_start",
    "gpu_load",
    "jitter_light",
    "bandwidth_20",
    "mixed",
    "cycle_end",
)

EXPERIMENT_PHASE_SET: frozenset[str] = frozenset(EXPERIMENT_PHASES)

#: The phase a cycle begins with. Collection starts when this phase is entered
#: from any other phase.
CYCLE_START_PHASE = "cycle_start"

#: The phase that ends a cycle and triggers result collection.
CYCLE_END_PHASE = "cycle_end"


def is_known_phase(phase: str) -> bool:
    """Return True when `phase` is part of the current scenario vocabulary."""
    return phase in EXPERIMENT_PHASE_SET
