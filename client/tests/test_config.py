"""Config parsing and defaults.

These defaults are what a group gets when they trim a value out of their .env,
so a drift between code, README, and .env.example changes results silently.
"""
import pytest


def test_documented_defaults(env) -> None:
    config = env()
    assert config.conf_threshold == 0.25
    assert config.frame_interval_ms == 100
    assert config.latency_deadline_ms == 300.0
    assert config.miss_penalty_px == 100.0
    assert config.sp_agent_interval_ms == 500
    assert config.sync_mode == "manual"
    assert config.initial_processing_mode == "local"


def test_latency_probes_default_on_for_every_agent(env) -> None:
    """
    The reference strategy recovers from local to remote using probe samples,
    and students are told to copy that idea. Probes defaulting off for
    SP_AGENT_CLASS=student left last_remote_probe_* as None forever, so a
    copied strategy silently never returned to remote.
    """
    assert env(SP_AGENT_CLASS="student").latency_probes_enabled is True
    assert env(SP_AGENT_CLASS="example").latency_probes_enabled is True


def test_latency_probes_can_still_be_disabled(env) -> None:
    assert env(LATENCY_PROBES_ENABLED="false").latency_probes_enabled is False


def test_target_class_filter_parsing(env) -> None:
    assert env(TARGET_CLASS_ID="").target_class_id is None
    assert env(TARGET_CLASS_ID="2").target_class_id == 2
    assert env(TARGET_CLASS_ID="2,7").target_class_id == (2, 7)
    assert env(TARGET_CLASS_ID=" 2 , 7 ").target_class_id == (2, 7)


def test_target_conf_threshold_falls_back_to_conf_threshold(env) -> None:
    config = env(CONFIDENCE_THRESHOLD="0.4")
    assert config.target_conf_threshold == 0.4
    config = env(CONFIDENCE_THRESHOLD="0.4", TARGET_CONFIDENCE_THRESHOLD="0.1")
    assert config.target_conf_threshold == 0.1


def test_by_phase_path_derived_from_results_path(env) -> None:
    config = env(RESULTS_LOG_PATH="/data/run.csv")
    assert config.results_by_phase_path == "/data/run_by_phase.csv"


def test_by_phase_path_can_be_set_explicitly(env) -> None:
    config = env(RESULTS_BY_PHASE_PATH="/data/custom.csv")
    assert config.results_by_phase_path == "/data/custom.csv"


def test_warmup_spike_phases_default_to_all(env) -> None:
    assert env().warmup_spike_phases == ()
    assert env(WARMUP_SPIKE_PHASES="gpu_load,mixed").warmup_spike_phases == (
        "gpu_load",
        "mixed",
    )


@pytest.mark.parametrize(
    "overrides, expected_message",
    [
        ({"SP_AGENT_CLASS": "nonsense"}, "SP_AGENT_CLASS"),
        ({"SYNC_MODE": "nonsense"}, "SYNC_MODE"),
        ({"INITIAL_PROCESSING_MODE": "nonsense"}, "INITIAL_PROCESSING_MODE"),
    ],
)
def test_invalid_values_name_the_variable(env, overrides, expected_message) -> None:
    """A typo in .env should say which variable is wrong, not raise generically."""
    with pytest.raises(ValueError, match=expected_message):
        env(**overrides)
