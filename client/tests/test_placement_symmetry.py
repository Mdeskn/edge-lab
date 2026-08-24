"""The baseline and the student's agent must be measured the same way.

Task 2.2 of the lab produces the `always-remote` baseline by clicking the
dashboard's Remote button, and students are then asked to beat that number with
their own agent. If a remote failure is handled differently depending on which
control asked for remote, part of every group's improvement is an artifact of
the harness rather than a better policy.
"""
import queue
import threading
import time

import numpy as np
import pytest

from shared_state import SharedState
from threads.dispatcher import Dispatcher
from threads.messages import FrameJob

FRAME = np.zeros((64, 64, 3), dtype=np.uint8)
PREDICTION = (10.0, 12.0, 5.0, 7.0, 15.0, 17.0)


class NoopSpan:
    def set_attribute(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class NoopTracer:
    def start_as_current_span(self, *_a, **_k):
        return NoopSpan()


class StubLocal:
    """Local inference that always succeeds, with a distinguishable answer."""

    def infer(self, _tensor, _shape):
        return PREDICTION


class StubRemote:
    """Remote inference that always fails, like a saturated or unreachable GPU."""

    def __init__(self, available: bool = True):
        self._available = available
        self.calls = 0

    def is_available(self) -> bool:
        return self._available

    def sends_raw_frames(self) -> bool:
        return True

    def infer(self, *_a, **_k):
        self.calls += 1
        raise RuntimeError("remote inference failed")


def run_one_frame(config, *, manual_lock: bool, remote_available: bool = True):
    """Dispatch a single frame and return (result_mode, prediction)."""
    state = SharedState(initial_mode="local")
    if manual_lock:
        state.lock_processing_mode("remote")
    else:
        state.set_processing_mode("remote")

    reader_queue: queue.Queue = queue.Queue()
    scorer_queue: queue.Queue = queue.Queue()
    dispatcher = Dispatcher(
        config, state, StubLocal(), StubRemote(available=remote_available),
        reader_queue, scorer_queue, NoopTracer(),
    )

    reader_queue.put(
        FrameJob(
            frame_number=1, video_cycle=0, frame=FRAME,
            gt_x=1.0, gt_y=1.0, enqueued_at=time.time(),
        )
    )

    thread = threading.Thread(target=dispatcher.run, daemon=True)
    thread.start()
    try:
        result = scorer_queue.get(timeout=5.0)
    finally:
        state.request_shutdown()
        thread.join(timeout=5.0)

    return result.mode, (result.predicted_x, result.predicted_y)


@pytest.mark.parametrize("remote_available", [True, False])
def test_manual_and_agent_remote_are_handled_identically(env, remote_available) -> None:
    """
    The same remote failure must produce the same outcome whether the dashboard
    locked the mode or the SP-Agent chose it.
    """
    config = env(REMOTE_FALLBACK_TO_LOCAL="true")

    manual_mode, manual_pred = run_one_frame(
        config, manual_lock=True, remote_available=remote_available
    )
    agent_mode, agent_pred = run_one_frame(
        config, manual_lock=False, remote_available=remote_available
    )

    assert manual_mode == agent_mode
    assert manual_pred == agent_pred


def test_failed_remote_falls_back_rather_than_scoring_a_miss(env) -> None:
    """
    With fallback on, a failed remote call produces a real local prediction.
    Under the old behaviour a manual lock returned (0, 0), which the Scorer
    treats as a missed detection and charges the full MISS_PENALTY_PX.
    """
    config = env(REMOTE_FALLBACK_TO_LOCAL="true")

    for manual_lock in (True, False):
        mode, prediction = run_one_frame(config, manual_lock=manual_lock)
        assert mode == "local_fallback", manual_lock
        assert prediction == PREDICTION[:2], manual_lock


def test_fallback_can_still_be_disabled_for_both_paths(env) -> None:
    """
    "Pure remote, no safety net" remains available. It is now a deliberate
    setting that applies to the baseline and the agent alike, rather than an
    implicit consequence of which control set the mode.
    """
    config = env(REMOTE_FALLBACK_TO_LOCAL="false")

    for manual_lock in (True, False):
        mode, prediction = run_one_frame(config, manual_lock=manual_lock)
        assert mode == "remote_unavailable", manual_lock
        assert prediction == (0.0, 0.0), manual_lock


def test_local_fallback_is_counted_separately_from_a_chosen_local(env) -> None:
    """
    Students can see how often the safety net fired: the per-phase table keeps
    local_fallback frames in their own column rather than folding them into
    local_frames.
    """
    state = SharedState()
    state.add_phase_result("gpu_load", 10.0, 100.0, 1.0, "local_fallback")
    state.add_phase_result("gpu_load", 10.0, 100.0, 1.0, "local")
    state.add_phase_result("gpu_load", 10.0, 100.0, 1.0, "remote")

    stats = state.get_phase_summary()["gpu_load"]
    assert stats["local_fallback_frames"] == 1
    assert stats["local_frames"] == 1
    assert stats["remote_frames"] == 1
