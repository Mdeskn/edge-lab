"""Incremental history delivery.

The backend used to resend the whole rolling window on every broadcast: about
260 KB of JSON twice a second per browser tab, 99% of it records the client
already had, serialized on the Pi whose inference latency is being graded.
"""
import json
import time

from dashboard.backend.state import SEQ_KEY, DashboardState


def make_state(max_history: int = 300) -> DashboardState:
    return DashboardState(max_history=max_history, group_id="1")


def add_frames(state: DashboardState, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        state.update_app_metric(
            {
                "frame_number": i,
                "timestamp": 1000.0 + i,
                "experiment_phase": "gpu_load",
                "processing_mode": "remote",
                "latency_ms": 100.0 + i,
                "displacement_px": 10.0,
                "cumulative_displacement_px": 10.0 * i,
            }
        )


def test_history_records_are_sequenced() -> None:
    state = make_state()
    add_frames(state, 3)
    seqs = [f[SEQ_KEY] for f in state.snapshot()["history"]["frames"]]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3


def test_sequence_is_shared_across_all_three_streams() -> None:
    """One cursor has to order frames, GPU, and network records together."""
    state = make_state()
    add_frames(state, 1)
    state.update_gpu_metrics({"gpu_utilization_pct": 50.0})
    state.update_network_metrics({"delay_ms": 5.0})

    snapshot = state.snapshot()
    seqs = [
        snapshot["history"]["frames"][0][SEQ_KEY],
        snapshot["history"]["gpu"][0][SEQ_KEY],
        snapshot["history"]["network"][0][SEQ_KEY],
    ]
    assert seqs == [1, 2, 3]
    assert snapshot["history_seq"] == 3


def test_full_snapshot_carries_the_whole_window() -> None:
    state = make_state()
    add_frames(state, 5)
    snapshot = state.snapshot()
    assert snapshot["history_mode"] == "full"
    assert len(snapshot["history"]["frames"]) == 5


def test_delta_carries_only_records_after_the_cursor() -> None:
    state = make_state()
    add_frames(state, 5)
    cursor = state.history_seq

    add_frames(state, 2, start=5)
    delta = state.snapshot(since_seq=cursor)

    assert delta["history_mode"] == "delta"
    assert [f["frame_number"] for f in delta["history"]["frames"]] == [5, 6]
    assert delta["history_seq"] == cursor + 2


def test_delta_is_empty_when_nothing_was_appended() -> None:
    state = make_state()
    add_frames(state, 3)
    delta = state.snapshot(since_seq=state.history_seq)
    assert delta["history"]["frames"] == []
    assert delta["history"]["gpu"] == []
    assert delta["history"]["network"] == []


def test_delta_records_stay_in_order() -> None:
    state = make_state()
    cursor = state.history_seq
    add_frames(state, 4)
    frames = state.snapshot(since_seq=cursor)["history"]["frames"]
    assert [f["frame_number"] for f in frames] == [0, 1, 2, 3]


def test_delta_after_the_window_rolled_returns_what_remains() -> None:
    """
    A cursor older than the whole retained window must not raise; the client
    receives everything still held and trims to max_history itself.
    """
    state = make_state(max_history=10)
    add_frames(state, 50)
    frames = state.snapshot(since_seq=0)["history"]["frames"]
    assert len(frames) == 10
    assert [f["frame_number"] for f in frames] == list(range(40, 50))


def test_delta_is_dramatically_smaller_than_a_full_window() -> None:
    state = make_state(max_history=300)
    add_frames(state, 300)
    for _ in range(300):
        state.update_gpu_metrics({"gpu_utilization_pct": 50.0})
        state.update_network_metrics({"delay_ms": 5.0})

    cursor = state.history_seq
    add_frames(state, 5, start=300)

    full = len(json.dumps(state.snapshot()))
    delta = len(json.dumps(state.snapshot(since_seq=cursor)))
    assert delta < full / 20, f"delta {delta} vs full {full}"


def test_history_endpoint_returns_the_seed_a_client_needs() -> None:
    state = make_state()
    add_frames(state, 3)
    history = state.history()
    assert set(history) == {"frames", "gpu", "network", "history_seq", "max_history"}
    assert history["history_seq"] == state.history_seq
    assert len(history["frames"]) == 3


def test_include_history_false_still_reports_the_cursor() -> None:
    state = make_state()
    add_frames(state, 3)
    snapshot = state.snapshot(include_history=False)
    assert snapshot["history_mode"] == "none"
    assert snapshot["history"]["frames"] == []
    assert snapshot["history_seq"] == 3


def test_excluded_frames_never_enter_history() -> None:
    """A frame left out of the score is left out of the charts too."""
    state = make_state()
    state.update_app_metric(
        {
            "frame_number": 1,
            "timestamp": time.time(),
            "latency_ms": 900.0,
            "displacement_px": 500.0,
            "excluded": True,
        }
    )
    assert state.snapshot()["history"]["frames"] == []
    assert state.snapshot()["summary"]["total_frames"] == 0
