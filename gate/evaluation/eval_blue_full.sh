#!/usr/bin/env bash
# Public BLUE closed-loop evaluation entry point.
# This script always runs the trained BLUE gate on Bench2Drive routes.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE_CKPT="${REPO_ROOT}/gate/weights/blue_simlingo_gate.pt"
GATE_THRESHOLD="0.66"
AGENT_CONFIG="${SIMLINGO_CKPT:-}"
CARLA_ROOT_ARG="${CARLA_ROOT:-}"
OUT_DIR="${REPO_ROOT}/outputs/blue_eval"
ROUTE_RANGE=""
SEED="1"
BASE_PORT="50000"
BASE_TM_PORT="50001"
TIMEOUT_SECONDS="4000"
HARD_TIMEOUT_SECONDS="4300"
KILL_AFTER_SECONDS="60"
MAX_RETRY_ROUNDS="1"
CONDA_ENV="simlingo"
ROUTE_DIR="${REPO_ROOT}/data/routes/bench2drive_split"
AGENT="${REPO_ROOT}/team_code/agent_simlingo.py"
FULL_ROUTES=($(seq 0 219))
ALL_ROUTES=("${FULL_ROUTES[@]}")

show_help() {
    cat <<EOF
BLUE closed-loop evaluation on Bench2Drive.

Usage:
  bash gate/evaluation/eval_blue_full.sh [options]

Required unless already set by environment:
  --agent-config PATH       SimLingo backbone pytorch_model.pt path.
  --carla-root PATH         CARLA root directory.

Options:
  --gate-ckpt PATH          BLUE gate checkpoint. Default: ${GATE_CKPT}
  --gate-threshold VALUE    Gate threshold. Default: ${GATE_THRESHOLD}
  --out-dir PATH            Output directory. Default: ${OUT_DIR}
  --route-range START:END   Python-style route slice, e.g. 0:1 for smoke test.
  --seed N                  Traffic manager seed. Default: ${SEED}
  --base-port N             First CARLA RPC port. Default: ${BASE_PORT}
  --timeout N               Leaderboard timeout in seconds. Default: ${TIMEOUT_SECONDS}
  --max-retry N             Retry rounds per route. Default: ${MAX_RETRY_ROUNDS}
  --conda-env NAME          Conda environment name. Default: ${CONDA_ENV}
  -h, --help                Show this help.

Examples:
  bash gate/evaluation/eval_blue_full.sh \
    --route-range 0:1 \
    --agent-config /path/to/pytorch_model.pt \
    --carla-root /path/to/carla \
    --out-dir outputs/blue_eval_smoke
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --gate-ckpt)
            GATE_CKPT="$2"
            shift 2
            ;;
        --gate-threshold)
            GATE_THRESHOLD="$2"
            shift 2
            ;;
        --agent-config)
            AGENT_CONFIG="$2"
            shift 2
            ;;
        --carla-root)
            CARLA_ROOT_ARG="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --route-range)
            ROUTE_RANGE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            BASE_TM_PORT="$((BASE_PORT + 1))"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            HARD_TIMEOUT_SECONDS="$((TIMEOUT_SECONDS + 300))"
            shift 2
            ;;
        --max-retry)
            MAX_RETRY_ROUNDS="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -n "${ROUTE_RANGE}" ]]; then
    RANGE_START="${ROUTE_RANGE%%:*}"
    RANGE_END="${ROUTE_RANGE##*:}"
    TOTAL="${#FULL_ROUTES[@]}"
    RANGE_START="${RANGE_START:-0}"
    RANGE_END="${RANGE_END:-${TOTAL}}"
    if [[ "${RANGE_START}" -ge "${TOTAL}" || "${RANGE_START}" -ge "${RANGE_END}" ]]; then
        echo "[ERROR] Invalid --route-range: ${ROUTE_RANGE}"
        exit 1
    fi
    if [[ "${RANGE_END}" -gt "${TOTAL}" ]]; then
        RANGE_END="${TOTAL}"
    fi
    ALL_ROUTES=("${FULL_ROUTES[@]:${RANGE_START}:$((RANGE_END - RANGE_START))}")
fi

if [[ -z "${AGENT_CONFIG}" ]]; then
    echo "[ERROR] --agent-config or SIMLINGO_CKPT is required."
    exit 1
fi
if [[ -z "${CARLA_ROOT_ARG}" ]]; then
    echo "[ERROR] --carla-root or CARLA_ROOT is required."
    exit 1
fi
if [[ ! -f "${GATE_CKPT}" ]]; then
    echo "[ERROR] BLUE gate checkpoint not found: ${GATE_CKPT}"
    exit 1
fi
if [[ ! -f "${AGENT_CONFIG}" ]]; then
    echo "[ERROR] SimLingo checkpoint not found: ${AGENT_CONFIG}"
    exit 1
fi
if [[ ! -d "${CARLA_ROOT_ARG}" ]]; then
    echo "[ERROR] CARLA root not found: ${CARLA_ROOT_ARG}"
    exit 1
fi
if [[ ! -d "${ROUTE_DIR}" ]]; then
    echo "[ERROR] Route directory not found: ${ROUTE_DIR}"
    exit 1
fi

for route_idx in "${ALL_ROUTES[@]}"; do
    file_num=$(printf '%02d' "${route_idx}")
    if [[ ! -f "${ROUTE_DIR}/bench2drive_${file_num}.xml" ]]; then
        echo "[ERROR] Route XML not found: ${ROUTE_DIR}/bench2drive_${file_num}.xml"
        exit 1
    fi
done

activate_environment() {
    echo "[BLUE] Activating conda environment: ${CONDA_ENV}"
    module load conda 2>/dev/null || true
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    else
        echo "[WARN] conda command not found; continuing with the current shell environment."
    fi
}

kill_carla_by_port() {
    local port="$1"
    pkill -f "CarlaUE4.*-carla-rpc-port=${port}" 2>/dev/null || true
    pkill -f "CarlaUE4.*-world-port=${port}" 2>/dev/null || true
    pkill -9 -f "CarlaUE4.*-carla-rpc-port=${port}" 2>/dev/null || true
    pkill -9 -f "CarlaUE4.*-world-port=${port}" 2>/dev/null || true
}

kill_all_carla() {
    for route_idx in "${ALL_ROUTES[@]}"; do
        local port=$((BASE_PORT + route_idx * 2))
        kill_carla_by_port "${port}"
    done
}

check_route_status() {
    local res_file="$1"
    if [[ ! -f "${res_file}" ]]; then
        echo "no"
        return
    fi
    python - "${res_file}" <<'PY' 2>/dev/null || true
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print("no")
    raise SystemExit(0)

records = data.get("_checkpoint", {}).get("records", [])
if not records:
    print("no")
    raise SystemExit(0)

for record in records:
    status = record.get("status", "")
    if not status or "Failed" in status:
        print("no")
        raise SystemExit(0)
print("yes")
PY
}

run_single_route() {
    local route_idx="$1"
    local file_num route_xml result_file port tm_port route_viz_dir
    file_num=$(printf '%02d' "${route_idx}")
    route_xml="${ROUTE_DIR}/bench2drive_${file_num}.xml"
    result_file="${OUT_DIR}/res/${file_num}_res.json"
    route_viz_dir="${OUT_DIR}/viz/${file_num}"
    port=$((BASE_PORT + route_idx * 2))
    tm_port=$((BASE_TM_PORT + route_idx * 2))

    mkdir -p "${OUT_DIR}/res" "${route_viz_dir}"
    export SAVE_PATH="${route_viz_dir}"
    export BLUE_MODE="trained_gate"
    export BLUE_GATE_CKPT="${GATE_CKPT}"
    export BLUE_GATE_THRESHOLD="${GATE_THRESHOLD}"
    export BLUE_OUTPUT_DIR="${OUT_DIR}"
    export SIMLINGO_CKPT="${AGENT_CONFIG}"
    export CARLA_ROOT="${CARLA_ROOT_ARG}"

    echo "[BLUE] Running route ${file_num} with trained_gate"
    timeout --foreground --signal=SIGTERM --kill-after="${KILL_AFTER_SECONDS}" "${HARD_TIMEOUT_SECONDS}s" \
    python -u "${REPO_ROOT}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" \
        --routes="${route_xml}" \
        --repetitions=1 \
        --track=SENSORS \
        --checkpoint="${result_file}" \
        --timeout="${TIMEOUT_SECONDS}" \
        --agent="${AGENT}" \
        --agent-config="${AGENT_CONFIG}" \
        --traffic-manager-seed="${SEED}" \
        --port="${port}" \
        --traffic-manager-port="${tm_port}"
    local eval_exit=$?
    echo "[BLUE] Route ${file_num} exited with code ${eval_exit}"
    kill_carla_by_port "${port}"
}

trap 'echo "[BLUE] Interrupted; cleaning up CARLA."; kill_all_carla; exit 1' INT TERM
trap 'kill_all_carla' EXIT

mkdir -p "${OUT_DIR}/res" "${OUT_DIR}/viz"
activate_environment
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:-}:${CARLA_ROOT_ARG}/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:${CARLA_ROOT_ARG}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/Bench2Drive/leaderboard"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}/Bench2Drive/scenario_runner"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}"
export SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

cat <<EOF
[BLUE] Evaluation configuration
  mode:            trained_gate
  gate checkpoint: ${GATE_CKPT}
  gate threshold:  ${GATE_THRESHOLD}
  agent config:    ${AGENT_CONFIG}
  carla root:      ${CARLA_ROOT_ARG}
  output dir:      ${OUT_DIR}
  seed:            ${SEED}
  route range:     ${ROUTE_RANGE:-all}
  routes:          ${ALL_ROUTES[*]}
EOF

for round in $(seq 1 "${MAX_RETRY_ROUNDS}"); do
    echo "[BLUE] Retry round ${round}/${MAX_RETRY_ROUNDS}"
    pending=0
    for route_idx in "${ALL_ROUTES[@]}"; do
        file_num=$(printf '%02d' "${route_idx}")
        result_file="${OUT_DIR}/res/${file_num}_res.json"
        if [[ "$(check_route_status "${result_file}")" == "yes" ]]; then
            echo "[BLUE] Route ${file_num} already finished; skipping."
            continue
        fi
        pending=$((pending + 1))
        run_single_route "${route_idx}" || true
    done
    if [[ "${pending}" -eq 0 ]]; then
        break
    fi
done

remaining=0
for route_idx in "${ALL_ROUTES[@]}"; do
    file_num=$(printf '%02d' "${route_idx}")
    result_file="${OUT_DIR}/res/${file_num}_res.json"
    if [[ "$(check_route_status "${result_file}")" != "yes" ]]; then
        remaining=$((remaining + 1))
    fi
done

if [[ "${remaining}" -eq 0 ]]; then
    echo "[BLUE] Evaluation completed successfully. Results: ${OUT_DIR}/res"
else
    echo "[BLUE] Evaluation finished with ${remaining} incomplete route(s). Results: ${OUT_DIR}/res"
    exit 2
fi
