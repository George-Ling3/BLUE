#!/usr/bin/env python3
"""Static smoke checks for BLUE Stage 1 assets."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    metadata = json.loads((REPO_ROOT / "gate/weights/blue_simlingo_gate.json").read_text())
    assert metadata["public_mode"] == "trained_gate"
    assert metadata["hidden_size"] == 896
    assert metadata["recommended_threshold"] == 0.66
    route_count = len(list((REPO_ROOT / "data/routes/bench2drive_split").glob("bench2drive_*.xml")))
    assert route_count == 220
    print("[OK] BLUE metadata and route assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
