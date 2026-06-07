#!/usr/bin/env python3
"""Placeholder downloader for the Stage 2 BLUE feature dataset."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the BLUE feature dataset when released.")
    parser.add_argument("--output-dir", default="data/external/blue-simlingo-gate-features-v1")
    parser.parse_args()
    print(
        "The Stage 2 feature dataset is not required for Stage 1 closed-loop "
        "evaluation. After release, this script will download "
        "blue-simlingo-gate-features-v1."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
