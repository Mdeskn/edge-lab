"""Scoring behaviour.

Every branch here changes a student's grade, so it is exercised directly
rather than only through a live run on the Pi.
"""
import io
import queue
import threading
import time

import numpy as np
import pytest

from shared_state import SharedState
from threads.messages import InferenceResult
from threads.scorer import Scorer

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class RecordingPublisher:
    """Captures what would have gone to Kafka / the dashboard."""

    def __init__(self):
        self.published: list[dict] = []

    def publish(self, **payload):
        self.published.append(payload)

    def publish_metric(self, **payload):
        self.published.append(payload)


def _drain(scorer: Scorer, scorer_queue: queue.Queue, state: SharedState) -> None:
    """Run the Scorer thread until it has consumed the queue, then stop it.

    The run loop checks for shutdown at the top, so it has to actually be
    running while the item is in flight rather than being single-stepped.
    """
    thread = threading.Thread(target=scorer.run, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while not scorer_queue.empty() and time.time() < deadline:
        time.sleep(0.005)
    # The item has been dequeued; give the body of the loop time to finish.
    time.sleep(0.05)
    state.request_shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "Scorer thread did not stop"


def make_result(**overrides) -> InferenceResult:
    defaults = {
        "frame_number": 10,
        "video_cycle": 0,
        "frame": FRAME,
        "gt_x": 100.0,
        "gt_y": 100.0,
        "predicted_x": 100.0,
        "predicted_y": 100.0,
        "predicted_x1": 90.0,
        "predicted_y1": 90.0,
        "predicted_x2": 110.0,
        "predicted_y2": 110.0,
        "latency_ms": 50.0,
        "mode": "local",
        "completed_at": time.time(),
    }
    defaults.update(overrides)
    return InferenceResult(**defaults)


@pytest.fixture
def scored(env):
    """Run one InferenceResult through a Scorer and return what it produced."""
    def _run(result: InferenceResult, *, gt=(100.0, 100.0), video_cycle=0, **config_overrides):
        config = env(**config_overrides)
        state = SharedState(initial_mode="local")
        state.update_ground_truth(result.frame_number, gt[0], gt[1], video_cycle)
        state.transition_to_armed()
        state.request_start_on_next_cycle()
        state.transition_to_collecting(time.time())

        kafka = RecordingPublisher()
        dashboard = RecordingPublisher()
        results_file = io.StringIO()
        scorer_queue: queue.Queue = queue.Queue()
        scorer = Scorer(config, state, scorer_queue, kafka, dashboard, results_file)

        scorer_queue.put(result)
        _drain(scorer, scorer_queue, state)

        return {
            "state": state,
            "scorer": scorer,
            "kafka": kafka.published,
            "csv": results_file.getvalue(),
            "summary": state.get_score_summary(),
        }

    return _run


def test_perfect_prediction_scores_zero_displacement(scored) -> None:
    out = scored(make_result(), gt=(100.0, 100.0))
    assert out["summary"]["frames_processed"] == 1
    assert out["summary"]["cumulative_displacement"] == pytest.approx(0.0)


def test_displacement_is_measured_against_current_ground_truth(scored) -> None:
    """
    Latency must cost displacement: the object keeps moving while inference
    runs, so the prediction is compared with where it is now, not with where
    it was at capture time.
    """
    out = scored(make_result(predicted_x=100.0, predicted_y=100.0), gt=(103.0, 104.0))
    assert out["summary"]["cumulative_displacement"] == pytest.approx(5.0)


def test_missed_detection_takes_the_fixed_penalty(scored) -> None:
    out = scored(
        make_result(predicted_x=0.0, predicted_y=0.0),
        MISS_PENALTY_PX="100.0",
    )
    assert out["summary"]["cumulative_displacement"] == pytest.approx(100.0)


def test_frame_without_capture_time_ground_truth_is_skipped(scored) -> None:
    out = scored(make_result(gt_x=None, gt_y=None))
    assert out["summary"]["frames_processed"] == 0


def test_video_wrap_frame_is_not_scored(scored) -> None:
    """
    If the clip looped while the frame was in flight, the current ground truth
    describes the start of the clip. Scoring against it would measure the
    video wrapping rather than the placement decision, and would punish slow
    remote frames hardest.
    """
    out = scored(make_result(video_cycle=0), gt=(1700.0, 170.0), video_cycle=1)
    assert out["summary"]["frames_processed"] == 0
    assert out["scorer"].excluded_counts["video_wrap"] == 1


def test_video_wrap_frame_is_flagged_downstream(scored) -> None:
    out = scored(make_result(video_cycle=0), video_cycle=1)
    assert out["kafka"][0]["excluded"] is True
    assert "excluded_video_wrap" in out["csv"].splitlines()[0]
    assert out["csv"].splitlines()[1].endswith(",0,1")


def test_same_cycle_frame_is_scored_normally(scored) -> None:
    out = scored(make_result(video_cycle=3), gt=(100.0, 100.0), video_cycle=3)
    assert out["summary"]["frames_processed"] == 1
    assert out["scorer"].excluded_counts["video_wrap"] == 0


def test_deadline_miss_is_recorded(scored) -> None:
    out = scored(make_result(latency_ms=400.0), LATENCY_DEADLINE_MS="300")
    assert out["kafka"][0]["deadline_miss"] is True
    phase_stats = out["state"].get_phase_summary()
    assert sum(s["deadline_misses"] for s in phase_stats.values()) == 1


def test_latency_reaches_kafka_unfiltered(scored) -> None:
    """The SP-Agent must see the real latency even for an excluded frame."""
    out = scored(make_result(latency_ms=987.0, video_cycle=0), video_cycle=1)
    assert out["kafka"][0]["latency_ms"] == pytest.approx(987.0)


def test_frames_are_not_scored_outside_collection(env) -> None:
    config = env()
    state = SharedState(initial_mode="local")
    state.update_ground_truth(10, 100.0, 100.0, 0)
    state.transition_to_armed()  # ARMED, never transitions to COLLECTING

    scorer_queue: queue.Queue = queue.Queue()
    results_file = io.StringIO()
    scorer = Scorer(
        config, state, scorer_queue,
        RecordingPublisher(), RecordingPublisher(), results_file,
    )
    scorer_queue.put(make_result(predicted_x=500.0, predicted_y=500.0))
    _drain(scorer, scorer_queue, state)

    assert state.get_score_summary()["frames_processed"] == 0
    # Header only: no data row is written outside a collection cycle.
    assert len(results_file.getvalue().strip().splitlines()) == 1


def test_placement_split_counts_fallback_separately(scored) -> None:
    out = scored(make_result(mode="local_fallback"))
    stats = next(iter(out["state"].get_phase_summary().values()))
    assert stats["local_fallback_frames"] == 1
    assert stats["local_frames"] == 0
    assert stats["remote_frames"] == 0
