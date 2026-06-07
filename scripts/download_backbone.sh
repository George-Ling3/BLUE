#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
BLUE uses the official SimLingo backbone checkpoint.

Before public release, replace TODO_OFFICIAL_SIMLINGO_BACKBONE_URL in
configs/assets.yaml with the official download URL. Place the checkpoint at:

  models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

or pass it explicitly to gate/evaluation/eval_blue_full.sh with --agent-config.
EOF
