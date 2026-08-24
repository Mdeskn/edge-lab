"""Guards on the API students are taught.

Both lab documents and the README show code that students copy verbatim. When
one of them drifts from the real API, the failure is quiet: a wrong metric key
returns its default instead of raising, so the strategy runs, the dashboard
updates, and a score comes out having ignored the signal entirely.
"""
import re
from pathlib import Path

import pytest

from common.gpu_metrics import GPU_METRIC_DEFAULTS, default_net_metrics
from student.sp_agent_base import SPAgentBase

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"


# --- The metric keys the documents teach must exist ----------------------


def published_metric_keys() -> set[str]:
    """Every key an agent can read off self.gpu_metrics or self.net_metrics."""
    return set(GPU_METRIC_DEFAULTS) | set(default_net_metrics()) | {"raw", "timestamp"}


@pytest.mark.parametrize(
    "key",
    ["gpu_util_pct", "gpu_mem_used_mb", "yolo_queue_ms", "total_pending"],
)
def test_gpu_keys_taught_to_students_exist(key) -> None:
    assert key in GPU_METRIC_DEFAULTS


@pytest.mark.parametrize("key", ["delay_ms", "jitter_ms", "packet_loss_pct"])
def test_network_keys_taught_to_students_exist(key) -> None:
    assert key in default_net_metrics()


@pytest.mark.parametrize(
    "key",
    [
        # Names that appeared in the lab documents but were never real. A
        # student using one of these silently reads 0 forever.
        "gpu_utilization_percent",
        "memory_used_mb",
        "gpu_utilization",
        "gpu_memory_used_mb",
        "network_delay_ms",
        "packet_loss_percent_",
    ],
)
def test_known_wrong_key_names_are_not_silently_valid(key) -> None:
    assert key not in GPU_METRIC_DEFAULTS


def test_readme_only_uses_real_metric_keys() -> None:
    """Every .get("key", ...) shown in the README must be a real key."""
    taught = set(re.findall(r'_metrics\.get\(\s*"([a-z0-9_]+)"', README.read_text()))
    assert taught, "no metric lookups found in the README; did the section move?"
    unknown = taught - published_metric_keys()
    assert not unknown, f"README teaches non-existent metric keys: {sorted(unknown)}"


# --- The decide() signature the documents teach --------------------------


def test_decide_takes_no_arguments_beyond_self() -> None:
    """
    The Instructions used to show decide(self, metrics). There is no metrics
    argument and no Metrics type; everything is read from self.
    """
    import inspect

    params = list(inspect.signature(SPAgentBase.decide).parameters)
    assert params == ["self"]


def test_readme_skeleton_matches_the_real_signature() -> None:
    skeleton = re.search(
        r"Minimal skeleton:\n\n```python\n(.*?)```", README.read_text(), re.S
    )
    assert skeleton, "README skeleton block not found"
    body = skeleton.group(1)
    assert "def decide(self) -> str:" in body
    assert "metrics:" not in body and "(self, metrics" not in body


def test_readme_skeleton_imports_resolve_inside_the_container() -> None:
    """
    The client runs with its own directory as the import root, so a `client.`
    prefix resolves from the repo root and fails on the Pi. This is the exact
    bug the README shipped with.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_sp_agent import offending_imports

    skeleton = re.search(
        r"Minimal skeleton:\n\n```python\n(.*?)```", README.read_text(), re.S
    ).group(1)
    assert offending_imports(skeleton) == []


def test_offending_imports_flags_the_container_breaking_prefix() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_sp_agent import offending_imports

    assert offending_imports("from client.student.sp_agent_base import SPAgentBase")
    assert offending_imports("import client.student.sp_agent_base")
    assert offending_imports("from student.sp_agent_base import SPAgentBase") == []


# --- The scoring description --------------------------------------------


def test_readme_does_not_describe_an_invented_scoring_formula() -> None:
    """
    The README carried `score = max(0, 1 - displacement / MISS_PENALTY_PX) x
    latency_factor`. No such formula exists: the score is summed displacement,
    and there is no latency_factor anywhere in the codebase.
    """
    text = README.read_text()
    assert "latency_factor" not in text
    assert "cumulative displacement" in text.lower()


# --- The client must start the way the docs say to start it ---------------


def test_client_starts_from_its_own_directory() -> None:
    """
    The container runs `python main.py` with the client directory as the
    working directory, and both the README and the lab runbook tell you to run
    it the same way from a checkout. Introducing the shared `common` package
    broke that: from client/ the repository root is not on sys.path, so every
    `common` import failed and the client died at startup, publishing nothing.

    Launching a real interpreter is the only honest check here, because the
    test session already has both directories on sys.path.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import main; print('ok')"],
        cwd=REPO_ROOT / "client",
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert result.returncode == 0, result.stderr
