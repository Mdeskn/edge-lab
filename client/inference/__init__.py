"""Inference backends: local ONNX and remote Triton."""
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient

__all__ = ["LocalServer", "RemoteClient"]
