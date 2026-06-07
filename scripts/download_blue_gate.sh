#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_PATH="${REPO_ROOT}/gate/weights/blue_simlingo_gate.pt"

if [[ -f "${GATE_PATH}" ]]; then
  echo "[OK] BLUE gate checkpoint already exists: ${GATE_PATH}"
  exit 0
fi

cat <<'EOF'
The Stage 1 package is expected to include gate/weights/blue_simlingo_gate.pt.
If you are using a source-only checkout, download the gate checkpoint from the
release URL listed in configs/assets.yaml and place it at gate/weights/.
EOF
exit 1
