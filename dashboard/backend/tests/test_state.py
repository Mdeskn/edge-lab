from dashboard.backend.state import DashboardState


def metric(timestamp: float = 1.0, mode: str = "local") -> dict:
    return {
        "timestamp": timestamp,
        "frame_number": 7,
        "experiment_phase": "cycle_start",
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


def test_excluded_metric_is_kept_out_of_history_and_summary() -> None:
    state = DashboardState(max_history=10, group_id="1")
    state.update_app_metric(metric() | {"excluded": True})

    snapshot = state.snapshot()
    assert snapshot["summary"]["total_frames"] == 0
    assert snapshot["summary"]["cumulative_displacement_px"] == 0.0
    assert len(snapshot["history"]["frames"]) == 0
    assert snapshot["latest"] == {}


def test_missing_infrastructure_metrics_are_empty() -> None:
    state = DashboardState(max_history=10, group_id="2")

    snapshot = state.snapshot()
    assert snapshot["group_id"] == "group2"
    assert snapshot["infrastructure"]["gpu"] == {}
    assert snapshot["infrastructure"]["network"] == {}
    assert snapshot["frame"]["url"] is None


def test_legacy_phase_topic_is_ignored_and_app_metric_sets_phase() -> None:
    state = DashboardState(max_history=10, group_id="1")
    state.update_phase("legacy_phase")

    state.update_app_metric(metric())

    snapshot = state.snapshot()
    assert snapshot["experiment_phase"] == "cycle_start"
    assert snapshot["latest"]["experiment_phase"] == "cycle_start"


def test_app_metrics_set_dashboard_phase_before_phase_topic() -> None:
    state = DashboardState(max_history=10, group_id="1")

    state.update_app_metric(metric())

    snapshot = state.snapshot()
    assert snapshot["experiment_phase"] == "cycle_start"


def test_latency_snapshot_keeps_local_and_remote_separate() -> None:
    state = DashboardState(max_history=10, group_id="1")
    state.update_app_metric(metric(timestamp=1.0, mode="remote") | {"latency_ms": 80.0})
    state.update_app_metric(metric(timestamp=2.0, mode="local") | {"latency_ms": 500.0})

    snapshot = state.snapshot()
    assert snapshot["latency"]["remote"]["latest_ms"] == 80.0
    assert snapshot["latency"]["remote"]["sample_count"] == 1
    assert snapshot["latency"]["local"]["latest_ms"] == 500.0
    assert snapshot["latency"]["local"]["sample_count"] == 1


def test_reset_preserves_latest_frame_but_clears_summary() -> None:
    state = DashboardState(max_history=10, group_id="3")
    state.update_frame(metric(), b"jpeg")

    state.reset()

    snapshot = state.snapshot()
    assert snapshot["summary"]["total_frames"] == 0
    assert snapshot["frame"]["url"] == "/api/video-stream"
    assert state.frame_image() == b"jpeg"


def test_frame_snapshot_returns_sequence_and_image() -> None:
    state = DashboardState(max_history=10, group_id="3")
    assert state.frame_snapshot() == (0, None)

    state.update_frame(metric(), b"first")
    state.update_frame(metric(timestamp=2.0), b"second")

    assert state.frame_snapshot() == (2, b"second")


def test_preview_drives_video_without_changing_score_summary() -> None:
    state = DashboardState(max_history=10, group_id="3")
    state.update_preview(
        {
            "frame_number": 9,
            "true_x": 100.0,
            "true_y": 200.0,
            "predicted_x": 90.0,
            "predicted_y": 190.0,
            "prediction_frame_number": 7,
        },
        b"preview",
    )

    snapshot = state.snapshot()
    assert snapshot["frame"]["frame_number"] == 9
    assert snapshot["frame"]["prediction_frame_number"] == 7
    assert snapshot["summary"]["total_frames"] == 0
    assert state.frame_snapshot() == (1, b"preview")
