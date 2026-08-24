"""Drive the real applyHistory() from app.js against the real server.

The history protocol has two halves: the backend decides what to put in a
delta, and the browser decides how to merge it. Testing only the Python half
missed a bug where the page's initial /api/state fetch raced the WebSocket
seed, replaced the buffer with older records, rewound the sequence cursor, and
left a permanent hole in the charts.

So this evaluates the actual JavaScript with node, fed by snapshots from the
actual DashboardState. It is skipped when node is unavailable.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.state import DashboardState

APP_JS = Path(__file__).resolve().parents[3] / "dashboard" / "frontend" / "static" / "app.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

# The self-contained history block in app.js, lifted by its boundary markers so
# the browser-only code around it never runs.
BLOCK_START = "const HISTORY_STREAMS"
BLOCK_END = "async function loadInitialState"

DRIVER = """
const steps = JSON.parse(process.argv[1]);
for (const step of steps) {
  applyHistory(step.state, { authoritative: step.authoritative });
}
console.log(JSON.stringify({
  frames: historyBuffers.frames.map((f) => f.frame_number),
  gpu: historyBuffers.gpu.length,
  seq: historySeq,
}));
"""


def run_merge(steps: list[dict]) -> dict:
    """Apply a sequence of snapshots through the real app.js merge function."""
    source = APP_JS.read_text()
    start = source.index(BLOCK_START)
    end = source.index(BLOCK_END)
    script = source[start:end] + DRIVER

    result = subprocess.run(
        ["node", "-e", script, json.dumps(steps)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def add(state: DashboardState, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        state.update_app_metric(
            {
                "frame_number": i,
                "timestamp": 1000.0 + i,
                "latency_ms": 100.0,
                "displacement_px": 10.0,
                "experiment_phase": "gpu_load",
                "processing_mode": "remote",
            }
        )


def test_socket_seed_then_deltas_reconstructs_the_stream() -> None:
    state = DashboardState(max_history=300, group_id="1")
    add(state, 10)
    steps = [{"state": state.snapshot(), "authoritative": True}]

    cursor = state.history_seq
    for i in range(10, 15):
        add(state, 1, start=i)
        steps.append(
            {"state": state.snapshot(since_seq=cursor), "authoritative": True}
        )
        cursor = state.history_seq

    assert run_merge(steps)["frames"] == list(range(15))


def test_late_api_state_fetch_does_not_punch_a_hole() -> None:
    """
    The page fetches /api/state and opens the WebSocket at the same time. When
    the socket wins, the slower fetch carries older history, and treating it as
    authoritative loses every record in between for the rest of the session.
    """
    state = DashboardState(max_history=300, group_id="1")
    add(state, 20)
    stale_fetch = state.snapshot()

    add(state, 5, start=20)
    cursor = state.history_seq
    socket_seed = state.snapshot()

    add(state, 3, start=25)
    delta = state.snapshot(since_seq=cursor)

    merged = run_merge(
        [
            {"state": socket_seed, "authoritative": True},
            {"state": stale_fetch, "authoritative": False},
            {"state": delta, "authoritative": True},
        ]
    )
    assert merged["frames"] == list(range(28))


def test_fetch_still_seeds_when_it_wins_the_race() -> None:
    """The fetch is not ignored outright; it seeds when the socket has not."""
    state = DashboardState(max_history=300, group_id="1")
    add(state, 12)

    merged = run_merge([{"state": state.snapshot(), "authoritative": False}])
    assert merged["frames"] == list(range(12))


def test_reconnect_replaces_rather_than_appends() -> None:
    """
    A socket reconnect sends a full window. It must replace, which is also how
    the page recovers if the backend restarted and its sequence began again.
    """
    first = DashboardState(max_history=300, group_id="1")
    add(first, 8)
    seed = first.snapshot()

    restarted = DashboardState(max_history=300, group_id="1")
    add(restarted, 3, start=100)

    merged = run_merge(
        [
            {"state": seed, "authoritative": True},
            {"state": restarted.snapshot(), "authoritative": True},
        ]
    )
    assert merged["frames"] == [100, 101, 102]


def test_overlapping_delta_does_not_duplicate() -> None:
    """
    The broadcast cursor is shared, so a client that connects mid-interval gets
    a first delta overlapping its seed.
    """
    state = DashboardState(max_history=300, group_id="1")
    add(state, 5)
    cursor = state.history_seq

    add(state, 5, start=5)
    seed = state.snapshot()
    overlapping = state.snapshot(since_seq=cursor)

    merged = run_merge(
        [
            {"state": seed, "authoritative": True},
            {"state": overlapping, "authoritative": True},
        ]
    )
    assert merged["frames"] == list(range(10))


def test_client_trims_to_the_servers_window() -> None:
    state = DashboardState(max_history=10, group_id="1")
    add(state, 5)
    steps = [{"state": state.snapshot(), "authoritative": True}]

    cursor = state.history_seq
    add(state, 20, start=5)
    steps.append({"state": state.snapshot(since_seq=cursor), "authoritative": True})

    merged = run_merge(steps)
    assert len(merged["frames"]) == 10
    assert merged["frames"] == list(range(15, 25))
