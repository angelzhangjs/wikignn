#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
PY_SCRIPT="$SCRIPT_DIR/export_to_pt.py"

GRAPH_DIR=""
OUT_PATH=""
FORMAT="basic"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --graph_dir DIR --out PATH [--format basic|pyg] [--python PY]

Options:
  --graph_dir DIR     Path to graph_out_dir with nodes/edges JSONL (required)
  --out PATH          Output .pt path (required)
  --format MODE       Output format: basic (default) or pyg
  --python PY         Python interpreter to use (default: python3 or \$PYTHON)
  -h, --help          Show this help and exit

Examples:
  $(basename "$0") --graph_dir /data/graph --out /tmp/graph.pt --format basic
  $(basename "$0") --graph_dir /data/graph --out /tmp/graph_pyg.pt --format pyg
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph_dir) GRAPH_DIR="${2:-}"; shift 2 ;;
    --out)       OUT_PATH="${2:-}"; shift 2 ;;
    --format)    FORMAT="${2:-}"; shift 2 ;;
    --python)    PYTHON="${2:-}"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${GRAPH_DIR}" || -z "${OUT_PATH}" ]]; then
  echo "Error: --graph_dir and --out are required." >&2
  usage
  exit 1
fi

if [[ ! -d "${GRAPH_DIR}" ]]; then
  echo "Error: graph_dir not found: ${GRAPH_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Error: export_to_pt.py not found at: ${PY_SCRIPT}" >&2
  exit 1
fi

if [[ "${FORMAT}" != "basic" && "${FORMAT}" != "pyg" ]]; then
  echo "Error: --format must be 'basic' or 'pyg' (got: ${FORMAT})" >&2
  exit 1
fi

# Activate local venv if present
if [[ -d "${SCRIPT_DIR}/.venv" && -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Quick check that torch is available in the chosen interpreter
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
  echo "Error: 'torch' not available in interpreter: ${PYTHON}" >&2
  echo "Hint: pip install torch (see PyTorch install docs) or pass --python to use a different interpreter." >&2
  exit 1
fi

echo "Executing:"
echo "  ${PYTHON} ${PY_SCRIPT} --graph_dir \"${GRAPH_DIR}\" --out \"${OUT_PATH}\" --format \"${FORMAT}\""
exec "${PYTHON}" "${PY_SCRIPT}" --graph_dir "${GRAPH_DIR}" --out "${OUT_PATH}" --format "${FORMAT}"


