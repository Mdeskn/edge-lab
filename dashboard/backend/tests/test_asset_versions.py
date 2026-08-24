"""The cache-busting version on the frontend assets must not go stale.

index.html references its assets with a ?v= parameter. Browsers key their
cache on the full URL, so an unchanged version string serves the old file no
matter what the server holds. Leaving it stale after changing the WebSocket
history protocol left browsers running JavaScript that could not merge deltas:
the charts rendered one delta at a time, about five points, and looked like
they were being overwritten rather than accumulating. Nothing in the server
logs showed a problem, because nothing was wrong on the server.
"""
import hashlib
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[3] / "dashboard" / "frontend" / "static"
INDEX = STATIC / "index.html"

VERSIONED = re.compile(r'(?:href|src)="/(app\.js|styles\.css)\?v=([^"]+)"')


def references() -> dict[str, str]:
    return {name: version for name, version in VERSIONED.findall(INDEX.read_text())}


@pytest.mark.parametrize("asset", ["app.js", "styles.css"])
def test_asset_is_referenced_with_a_version(asset) -> None:
    found = references()
    assert asset in found, (
        f"{asset} is referenced without a ?v= cache buster, so browsers can "
        f"keep serving an old copy after a deploy"
    )


def test_both_assets_share_one_version_string() -> None:
    """One version for the bundle keeps the two from drifting apart."""
    versions = set(references().values())
    assert len(versions) == 1, f"assets disagree on version: {references()}"


def test_version_matches_the_current_asset_contents() -> None:
    """
    A fingerprint of the shipped assets, recorded next to index.html. When
    app.js or styles.css changes without the version being bumped, this fails
    and names what to do. Regenerate by updating the ?v= string and running:

        python -c "import hashlib,pathlib; \\
            d=pathlib.Path('dashboard/frontend/static'); \\
            print(hashlib.sha256(b''.join(sorted((d/n).read_bytes() \\
            for n in ('app.js','styles.css')))).hexdigest()[:12])"
    """
    stamp = STATIC / ".asset-version"
    digest = hashlib.sha256(
        b"".join(sorted((STATIC / n).read_bytes() for n in ("app.js", "styles.css")))
    ).hexdigest()[:12]
    version = next(iter(references().values()))

    if not stamp.exists():
        stamp.write_text(f"{version} {digest}\n")
        pytest.skip("recorded the initial asset fingerprint")

    recorded_version, recorded_digest = stamp.read_text().split()
    if digest == recorded_digest:
        assert version == recorded_version, (
            "assets are unchanged but the version string moved; revert it or "
            f"update {stamp.name}"
        )
        return

    assert version != recorded_version, (
        "app.js or styles.css changed but index.html still says "
        f"?v={version}. Browsers will keep serving the cached copy. Bump the "
        f"version in index.html and update {stamp.name} to:\n"
        f"    <new-version> {digest}"
    )
    stamp.write_text(f"{version} {digest}\n")
