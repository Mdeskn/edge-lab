"""Shared fixtures for the client test suite."""
import os

import pytest

from config import load_config

#: The environment variables load_config() requires with no default.
REQUIRED_ENV = {
    "VIDEO_PATH": "/data/test_video.mp4",
    "GROUND_TRUTH_PATH": "/data/ground_truth.csv",
    "MODEL_PATH": "/data/yolov10n.onnx",
}


@pytest.fixture
def env(monkeypatch):
    """
    Build a Config from a controlled environment.

    load_config() reads os.environ directly, so every test that needs a Config
    goes through here rather than constructing the dataclass by hand, which
    would silently drift from the real defaults being tested.
    """
    def _build(**overrides):
        # Clear anything the developer's own .env put in the environment so a
        # local shell cannot change what the defaults resolve to.
        for key in list(os.environ):
            if key.isupper() and ("_" in key or key in ("GROUP_ID",)):
                monkeypatch.delenv(key, raising=False)
        for key, value in {**REQUIRED_ENV, **overrides}.items():
            monkeypatch.setenv(key, str(value))
        return load_config()

    return _build
