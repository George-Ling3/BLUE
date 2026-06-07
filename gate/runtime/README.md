# BLUE Gate Runtime

The runtime contains small utilities for BLUE gate inference and decision logs.
The closed-loop path writes:

- `blue_gate_decisions.jsonl`: one JSON object per evaluated frame.
- `blue_eval_summary.json`: route-level gate counts and summary statistics.

A gate decision of `1` means generate language before producing the driving action.
A gate decision of `0` means skip language generation and directly produce the driving action.
