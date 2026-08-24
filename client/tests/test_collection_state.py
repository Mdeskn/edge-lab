"""The collection state machine and placement locking in SharedState."""
import time

import pytest

from shared_state import CollectionState, SharedState


def armed() -> SharedState:
    state = SharedState(initial_mode="local")
    state.transition_to_armed()
    return state


# --- Placement -----------------------------------------------------------


def test_invalid_placement_is_rejected() -> None:
    state = SharedState()
    with pytest.raises(ValueError, match="local"):
        state.set_processing_mode("gpu")


def test_manual_lock_blocks_then_releases_agent_control() -> None:
    """
    A manual override used to be a one-way door: nothing called
    unlock_processing_mode and the UI had no Auto button, so one click
    disabled the SP-Agent for the rest of the run.
    """
    state = SharedState(initial_mode="local")
    assert state.is_processing_mode_locked() is False

    state.lock_processing_mode("remote")
    assert state.is_processing_mode_locked() is True
    assert state.get_processing_mode() == "remote"

    state.unlock_processing_mode()
    assert state.is_processing_mode_locked() is False
    state.set_processing_mode("local")
    assert state.get_processing_mode() == "local"


# --- Cycle lifecycle -----------------------------------------------------


def test_starts_disconnected() -> None:
    assert SharedState().get_collection_state() == CollectionState.DISCONNECTED


def test_collecting_requires_an_explicit_start_request() -> None:
    state = armed()
    assert state.transition_to_collecting(time.time()) is False
    assert state.get_collection_state() == CollectionState.ARMED

    state.request_start_on_next_cycle()
    assert state.transition_to_collecting(time.time()) is True
    assert state.is_collecting() is True


def test_start_request_is_ignored_unless_armed() -> None:
    state = SharedState()  # DISCONNECTED
    state.request_start_on_next_cycle()
    assert state.transition_to_collecting(time.time()) is False


def test_second_transition_to_collecting_is_refused() -> None:
    state = armed()
    state.request_start_on_next_cycle()
    assert state.transition_to_collecting(time.time()) is True
    assert state.transition_to_collecting(time.time()) is False


def test_complete_records_the_final_score_and_counts_the_cycle() -> None:
    state = armed()
    state.request_start_on_next_cycle()
    state.transition_to_collecting(time.time())
    state.transition_to_complete(time.time(), 1234.5)

    snapshot = state.get_collection_snapshot()
    assert snapshot["state"] == "complete"
    assert snapshot["final_cumulative_displacement"] == 1234.5
    assert snapshot["cycles_completed"] == 1


def test_rearming_clears_per_cycle_totals_but_keeps_cycle_count() -> None:
    state = armed()
    state.request_start_on_next_cycle()
    state.transition_to_collecting(time.time())
    state.add_displacement(500.0)
    state.add_phase_result("gpu_load", 500.0, 90.0, 3.0, "remote")
    state.transition_to_complete(time.time(), 500.0)

    state.transition_to_armed()

    assert state.get_score_summary()["cumulative_displacement"] == 0.0
    assert state.get_score_summary()["frames_processed"] == 0
    assert state.get_phase_summary() == {}
    assert state.get_collection_snapshot()["cycles_completed"] == 1


def test_phases_seen_only_accumulate_while_collecting() -> None:
    state = armed()
    state.add_phase_seen("gpu_load")
    assert state.get_collection_snapshot()["phases_seen"] == []

    state.request_start_on_next_cycle()
    state.transition_to_collecting(time.time())
    state.add_phase_seen("gpu_load")
    assert "gpu_load" in state.get_collection_snapshot()["phases_seen"]


# --- Scoring aggregates --------------------------------------------------


def test_score_summary_averages_over_frames() -> None:
    state = SharedState()
    state.add_displacement(10.0)
    state.add_displacement(20.0)
    summary = state.get_score_summary()
    assert summary["cumulative_displacement"] == 30.0
    assert summary["frames_processed"] == 2
    assert summary["average_displacement"] == 15.0


def test_phase_result_ignores_unscoreable_frames() -> None:
    state = SharedState()
    state.add_phase_result("gpu_load", None, 90.0, 3.0, "remote")
    assert state.get_phase_summary() == {}


def test_ground_truth_round_trips_the_video_cycle() -> None:
    state = SharedState()
    state.update_ground_truth(42, 1.0, 2.0, 3)
    assert state.get_ground_truth() == (42, 1.0, 2.0, 3)
