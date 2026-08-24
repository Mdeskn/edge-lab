"""Saving charts from a finished cycle.

Students paste these into the presentation the task sheet asks for, so the
saved file has to stand on its own outside the dashboard.
"""
import base64
import struct
import zlib
from pathlib import Path

import pytest
from fastapi import HTTPException

from dashboard.backend.main import _write_chart

APP_JS = Path(__file__).resolve().parents[3] / "dashboard" / "frontend" / "static" / "app.js"


def tiny_png() -> bytes:
    """A valid 1x1 PNG, built here so the test needs no image library."""
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


def test_base64_png_is_saved_as_png(tmp_path) -> None:
    dest = _write_chart(base64.b64encode(tiny_png()).decode(), tmp_path / "chart", "latency")
    assert dest.suffix == ".png"
    assert dest.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_svg_markup_is_still_accepted(tmp_path) -> None:
    """An older frontend sends SVG; it should save rather than fail."""
    dest = _write_chart('<svg xmlns="http://www.w3.org/2000/svg"></svg>', tmp_path / "c", "latency")
    assert dest.suffix == ".svg"
    assert dest.read_text().startswith("<svg")


def test_garbage_is_rejected_rather_than_written(tmp_path) -> None:
    with pytest.raises(HTTPException) as excinfo:
        _write_chart("not base64 and not svg!!", tmp_path / "c", "latency")
    assert excinfo.value.status_code == 400
    assert "latency" in excinfo.value.detail
    assert not list(tmp_path.iterdir())


def test_base64_of_something_that_is_not_a_png_is_rejected(tmp_path) -> None:
    with pytest.raises(HTTPException) as excinfo:
        _write_chart(base64.b64encode(b"GIF89a").decode(), tmp_path / "c", "placement")
    assert excinfo.value.status_code == 400
    assert "not a PNG" in excinfo.value.detail


# --- The exported image must not depend on the page stylesheet -----------


def test_exported_svg_inlines_the_styles_it_uses() -> None:
    """
    The on-page charts get their axis and label styling from styles.css. An
    exported file carries no stylesheet, so every class the markup uses must be
    defined in SAVED_CHART_STYLE. Missing ones silently lose their stroke: the
    axis and gridlines render invisible and labels fall back to black.
    """
    source = APP_JS.read_text()
    style_block = source[
        source.index("const SAVED_CHART_STYLE") : source.index("const SAVED_CHART_SCALE")
    ]

    used = {
        "chart-axis",
        "chart-gridline",
        "chart-band-a",
        "chart-band-b",
        "chart-band-boundary",
        "chart-band-label",
        "chart-tick-label",
        "chart-end-ring",
        "chart-end-label",
        "chart-crosshair",
    }
    missing = {name for name in used if f".{name}" not in style_block}
    assert not missing, f"exported charts would lose styling for: {sorted(missing)}"


def test_exported_svg_is_self_contained() -> None:
    """A rasteriser needs the namespace and explicit dimensions."""
    source = APP_JS.read_text()
    wrapper = source[
        source.index("function standaloneChartSvg") : source.index("function chartSvgToPngBase64")
    ]
    assert 'xmlns="http://www.w3.org/2000/svg"' in wrapper
    assert "width=" in wrapper and "height=" in wrapper
    assert 'fill="#ffffff"' in wrapper, "needs an opaque background for slides"
