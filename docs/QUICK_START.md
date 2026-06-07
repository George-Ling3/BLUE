# BLUE Quick Start

This guide covers Stage 1: closed-loop Bench2Drive evaluation with the trained
BLUE gate.

## 1. Create the environment

```bash
module load conda
conda create -n simlingo python=3.8 -y
conda activate simlingo
python -m pip install -r requirements.txt
```

Install CARLA 0.9.15 and make sure the CARLA root directory contains `CarlaUE4.sh`
and `PythonAPI/carla`.

## 2. Prepare checkpoints

BLUE expects two checkpoints:

1. Official SimLingo backbone checkpoint.
2. BLUE gate checkpoint at `gate/weights/blue_simlingo_gate.pt`.

Verify the gate checkpoint:

```bash
python scripts/verify_assets.py
```

## 3. Run a one-route smoke test

```bash
bash gate/evaluation/eval_blue_full.sh \
  --route-range 0:1 \
  --agent-config /path/to/pytorch_model.pt \
  --carla-root /path/to/carla \
  --out-dir outputs/blue_eval_smoke
```

The script always runs `trained_gate`. It does not expose other evaluation modes.

## 4. Run the full Bench2Drive split

```bash
bash gate/evaluation/eval_blue_full.sh \
  --agent-config /path/to/pytorch_model.pt \
  --carla-root /path/to/carla \
  --out-dir outputs/blue_eval_full
```

For multi-machine evaluation, split routes with `--route-range START:END`.

## 5. Outputs

- Standard Bench2Drive results: `outputs/blue_eval_*/res/`.
- Route visualizations and metrics: `outputs/blue_eval_*/viz/`.
- Gate decisions: `blue_gate_decisions.jsonl`.
- Gate summary: `blue_eval_summary.json`.
