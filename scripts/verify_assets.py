#!/usr/bin/env python3
"""Verify BLUE release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_FILES = [
    "gate/weights/blue_simlingo_gate.pt",
    "gate/weights/blue_simlingo_gate.json",
    "gate/weights/SHA256SUMS",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify BLUE Stage 1 assets.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    missing = [rel for rel in DEFAULT_FILES if not (repo_root / rel).is_file()]
    if missing:
        for rel in missing:
            print(f"[ERROR] Missing asset: {rel}")
        return 1

    metadata_path = repo_root / "gate/weights/blue_simlingo_gate.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("sha256")
    actual = sha256(repo_root / "gate/weights/blue_simlingo_gate.pt")
    if expected != actual:
        print(f"[ERROR] Gate checksum mismatch: expected={expected}, actual={actual}")
        return 2

    sums_path = repo_root / "gate/weights/SHA256SUMS"
    sums_text = sums_path.read_text(encoding="utf-8")
    if actual not in sums_text:
        print("[ERROR] SHA256SUMS does not contain the gate checksum.")
        return 3

    route_count = len(list((repo_root / "data/routes/bench2drive_split").glob("bench2drive_*.xml")))
    if route_count != 220:
        print(f"[ERROR] Expected 220 Bench2Drive routes, found {route_count}.")
        return 4

    print("[OK] BLUE Stage 1 assets verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
