from spike_filter import WarmupSpikeFilter


def make_filter(**overrides) -> WarmupSpikeFilter:
    params = dict(enabled=True, settle_sec=2.0, multiplier=3.0, floor_ms=150.0)
    params.update(overrides)
    return WarmupSpikeFilter(**params)


def establish_baseline(f: WarmupSpikeFilter, phase: str, latency_ms: float, start: float) -> None:
    for i in range(10):
        f.check(phase, latency_ms, now=start + i)


def test_no_exclusion_outside_target_phase() -> None:
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
