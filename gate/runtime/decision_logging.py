"""Utilities for BLUE gate decision logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union


def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Union[str, Path], records: Iterable[Dict[str, Any]], threshold: float) -> None:
    """Write a compact summary from gate decision records."""
    items = list(records)
    generate = sum(1 for item in items if int(item.get("gate_decision", 0)) == 1)
    direct = len(items) - generate
    summary = {
        "threshold": threshold,
        "stats": {"generate": generate, "direct": direct},
        "generate_ratio": generate / len(items) if items else 0.0,
        "total_steps": len(items),
        "decision_log": items,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
