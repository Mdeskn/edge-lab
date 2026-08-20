"""Typed payloads passed between the pipeline threads.

These used to be bare tuples unpacked positionally in three places, so adding
a field meant editing every call site and any missed one failed at runtime
with an unpacking error. Named fields also let the Scorer read
`result.video_cycle` instead of counting commas.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FrameJob:
    """One captured frame on its way from the FrameReader to the Dispatcher."""

    frame_number: int
    #: How many times the clip has looped. Travels with the frame so the
    #: Scorer can tell whether the video wrapped while this frame was in
    #: flight.
    video_cycle: int
    frame: np.ndarray
    gt_x: float | None
    gt_y: float | None
    enqueued_at: float


@dataclass(slots=True)
class InferenceResult:
    """One completed inference on its way from the Dispatcher to the Scorer."""

    frame_number: int
    video_cycle: int
    frame: np.ndarray
    #: Ground truth at capture time. Only gates whether the frame is scored.
    gt_x: float | None
    gt_y: float | None
    predicted_x: float
    predicted_y: float
    predicted_x1: float
    predicted_y1: float
    predicted_x2: float
    predicted_y2: float
    latency_ms: float
    #: "local", "remote", "local_fallback", or "remote_unavailable".
    mode: str
    completed_at: float

    @property
    def has_prediction(self) -> bool:
        """Return True when the backend returned a box rather than a miss."""
        return self.predicted_x != 0.0 or self.predicted_y != 0.0
