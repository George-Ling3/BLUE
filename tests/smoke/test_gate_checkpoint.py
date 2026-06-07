#!/usr/bin/env python3
"""CPU smoke test for the BLUE gate checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simlingo_training.models.gate import create_gate  # noqa: E402


def main() -> int:
    gate_path = REPO_ROOT / "gate/weights/blue_simlingo_gate.pt"
    gate = create_gate("trained_gate", hidden_size=896, ckpt_path=str(gate_path), threshold=0.66)
    gate.eval()
    with torch.no_grad():
        decisions, scores = gate.forward_with_prob(torch.zeros(2, 896))
    assert decisions.shape == (2,)
    assert scores.shape == (2,)
    assert decisions.dtype == torch.long
    print("[OK] BLUE gate checkpoint loads and runs on CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
