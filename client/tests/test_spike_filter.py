from spike_filter import WarmupSpikeFilter


def make_filter(**overrides) -> WarmupSpikeFilter:
    params = {"enabled": True, "settle_sec": 2.0, "multiplier": 3.0, "floor_ms": 150.0}
    params.update(overrides)
    return WarmupSpikeFilter(**params)


def establish_baseline(f: WarmupSpikeFilter, phase: str, latency_ms: float, start: float) -> None:
    for i in range(10):
        f.check(phase, latency_ms, now=start + i)


def test_no_exclusion_without_a_transition() -> None:
    f = make_filter()
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("cycle_start", 900.0, now=10.0) is False


def test_no_exclusion_without_baseline() -> None:
    f = make_filter()
    assert f.check("gpu_load", 900.0, now=0.0) is False


def test_first_big_spike_on_phase_entry_is_excluded() -> None:
    f = make_filter()
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("gpu_load", 400.0, now=10.0) is True


def test_normal_latency_after_transition_is_not_excluded() -> None:
    f = make_filter()
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("gpu_load", 400.0, now=10.0) is True
    assert f.check("gpu_load", 55.0, now=10.1) is False


def test_only_one_exclusion_per_phase_entry() -> None:
    f = make_filter()
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("gpu_load", 400.0, now=10.0) is True
    assert f.check("gpu_load", 500.0, now=10.2) is False


def test_spike_after_settle_window_is_not_excluded() -> None:
    f = make_filter(settle_sec=2.0)
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("gpu_load", 45.0, now=10.0) is False
    assert f.check("gpu_load", 900.0, now=13.0) is False


def test_reentering_phase_rearms_exclusion() -> None:
    f = make_filter()
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("gpu_load", 400.0, now=10.0) is True
    establish_baseline(f, "jitter_light", 45.0, start=20.0)
    assert f.check("gpu_load", 400.0, now=30.0) is True


def test_disabled_filter_never_excludes() -> None:
    f = make_filter(enabled=False)
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("gpu_load", 900.0, now=10.0) is False


# --- Applying to every phase, not just gpu_load ---------------------------


def test_every_phase_transition_arms_by_default() -> None:
    """An identical artifact must be treated identically in every phase."""
    for phase in ("jitter_light", "bandwidth_20", "mixed", "cycle_end"):
        f = make_filter()
        establish_baseline(f, "cycle_start", 40.0, start=0.0)
        assert f.check(phase, 400.0, now=10.0) is True, phase


def test_phase_allowlist_restricts_which_transitions_arm() -> None:
    f = make_filter(phases=("gpu_load",))
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("jitter_light", 400.0, now=10.0) is False

    f = make_filter(phases=("gpu_load",))
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    assert f.check("gpu_load", 400.0, now=10.0) is True


def test_window_closes_when_moving_into_an_uncovered_phase() -> None:
    """A covered phase's window must not leak into the next phase's samples."""
    f = make_filter(phases=("gpu_load",), settle_sec=5.0)
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    # Enter the covered phase without spending the exclusion.
    assert f.check("gpu_load", 45.0, now=10.0) is False
    # Leaving it while the window is still open must close the window.
    assert f.check("jitter_light", 900.0, now=11.0) is False


def test_below_floor_spike_is_not_excluded() -> None:
    """A spike must clear both the multiplier and the absolute floor."""
    f = make_filter(floor_ms=150.0)
    establish_baseline(f, "cycle_start", 10.0, start=0.0)
    # 40 ms is 4x the 10 ms baseline but well under the 150 ms floor.
    assert f.check("gpu_load", 40.0, now=10.0) is False


def test_excluded_count_tracks_flagged_samples() -> None:
    f = make_filter()
    assert f.excluded_count == 0
    establish_baseline(f, "cycle_start", 40.0, start=0.0)
    f.check("gpu_load", 400.0, now=10.0)
    assert f.excluded_count == 1
    establish_baseline(f, "mixed", 40.0, start=20.0)
    f.check("gpu_load", 400.0, now=30.0)
    assert f.excluded_count == 2
