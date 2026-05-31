from dashboard.backend.state import DashboardState


def metric(timestamp: float = 1.0, mode: str = "local") -> dict:
    return {
        "timestamp": timestamp,
        "frame_number": 7,
        "experiment_phase": "baseline",
        "processing_mode": mode,
        "latency_ms": 12.0,
        "displacement_px": 4.0,
        "cumulative_displacement_px": 9.0,
    }


def test_duplicate_metric_does_not_double_count_summary() -> None:
    state = DashboardState(max_history=10, group_id="1")
    state.update_app_metric(metric())
    state.update_app_metric(metric())

    snapshot = state.snapshot()
    assert snapshot["group_id"] == "group1"
    assert snapshot["summary"]["total_frames"] == 1
    assert snapshot["summary"]["local_frames"] == 1
    assert len(snapshot["history"]["frames"]) == 1


def test_missing_infrastructure_metrics_are_empty() -> None:
    state = DashboardState(max_history=10, group_id="2")

    snapshot = state.snapshot()
    assert snapshot["group_id"] == "group2"
    assert snapshot["infrastructure"]["gpu"] == {}
    assert snapshot["infrastructure"]["network"] == {}
    assert snapshot["frame"]["url"] is None


def test_reset_preserves_latest_frame_but_clears_summary() -> None:
    state = DashboardState(max_history=10, group_id="3")
    state.update_frame(metric(), b"jpeg")

    state.reset()

    snapshot = state.snapshot()
    assert snapshot["summary"]["total_frames"] == 0
    assert snapshot["frame"]["url"] == "/api/frame?v=1"
    assert state.frame_image() == b"jpeg"
