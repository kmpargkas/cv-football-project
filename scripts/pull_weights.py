"""Fetch and verify model weights listed in weights/manifest.json.

Weight files are git-ignored, so this is how a fresh clone gets them.
Entries without a URL are reported as missing with a hint rather than fetched.

Usage:
    uv run python scripts/pull_weights.py           # fetch anything missing
    uv run python scripts/pull_weights.py --list    # just show the registry
    uv run python scripts/pull_weights.py --verify  # check checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
MANIFEST = WEIGHTS_DIR / "manifest.json"


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"No manifest at {MANIFEST}")
    data = json.loads(MANIFEST.read_text())
    return data.get("weights", [])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_ok(entry: dict, path: Path) -> bool | None:
    """True/False against the manifest sha256, None when the entry has none."""
    expected = entry.get("sha256")
    if not expected:
        return None
    return sha256(path) == expected


def cmd_list(entries: list[dict]) -> None:
    if not entries:
        print("Manifest is empty — no weights registered yet.")
        return
    for e in entries:
        path = WEIGHTS_DIR / e["file"]
        status = "present" if path.exists() else "MISSING"
        print(f"[{status:>7}] {e.get('component', '?'):<10} {e['file']}")
        if e.get("metrics"):
            print(f"            metrics: {e['metrics']}")


def cmd_verify(entries: list[dict]) -> int:
    failures = 0
    for e in entries:
        path = WEIGHTS_DIR / e["file"]
        if not path.exists():
            print(f"MISSING  {e['file']}")
            failures += 1
        elif _checksum_ok(e, path) is False:
            print(f"CORRUPT  {e['file']} (checksum mismatch)")
            failures += 1
        else:
            print(f"OK       {e['file']}")
    return failures


def download(url: str, dest: Path, timeout: int = 60) -> None:
    """Download to a temp file, then move into place, so failures leave no partial file."""
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Refusing non-HTTP(S) URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        tmp.write_bytes(response.read())
    tmp.replace(dest)


def cmd_pull(entries: list[dict]) -> int:
    failures = 0
    for e in entries:
        path = WEIGHTS_DIR / e["file"]
        if path.exists():
            print(f"SKIP     {e['file']} (already present)")
            continue
        url = e.get("url")
        if not url:
            print(
                f"MANUAL   {e['file']} — no download URL in the manifest; "
                f"place it at weights/{e['file']}"
            )
            failures += 1
            continue

        print(f"FETCH    {e['file']} <- {url}")
        try:
            download(url, path)
        except Exception as exc:  # report and continue to the next entry
            print(f"FAILED   {e['file']}: {exc}")
            failures += 1
            continue

        ok = _checksum_ok(e, path)
        if ok is None:
            print(f"WARN     {e['file']} downloaded but has no sha256 in the manifest")
        elif ok:
            print(f"VERIFIED {e['file']}")
        else:
            path.unlink()
            print(f"CORRUPT  {e['file']} (checksum mismatch, deleted)")
            failures += 1
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch/verify weights from the manifest.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="Show the registry and exit.")
    mode.add_argument("--verify", action="store_true", help="Verify checksums of present files.")
    args = p.parse_args()

    entries = load_manifest()
    if args.list:
        cmd_list(entries)
        return
    failures = cmd_verify(entries) if args.verify else cmd_pull(entries)
    if failures:
        sys.exit(f"{failures} weight(s) need attention.")


if __name__ == "__main__":
    main()
