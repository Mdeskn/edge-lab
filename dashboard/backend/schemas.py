"""Validated payloads accepted by the dashboard backend."""
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AppMetric(BaseModel):
    """A scored frame record published by the client or Kafka."""

    model_config = ConfigDict(extra="allow")

    timestamp: float = Field(default_factory=time.time)
    frame_number: int
    group_id: str | None = None
    experiment_phase: str = "unknown"
    processing_mode: str = "unknown"
    latency_ms: float | None = None
    displacement_px: float | None = None
    true_x: float | None = None
    true_y: float | None = None
    predicted_x: float | None = None
    predicted_y: float | None = None
    cumulative_displacement_px: float | None = None


class FrameUpdate(AppMetric):
    """A scored frame record with an annotated JPEG encoded as base64."""

    image_base64: str


class PhaseMetric(BaseModel):
    """Current experiment phase."""

    model_config = ConfigDict(extra="allow")

    timestamp: float = Field(default_factory=time.time)
    phase: str = "unknown"


class PlacementControlRequest(BaseModel):
    """Manual local/remote placement command from the dashboard."""

    mode: Literal["local", "remote"]
