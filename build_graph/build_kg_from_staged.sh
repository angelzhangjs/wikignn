#!/usr/bin/env bash
set -euo pipefail

# Build a heterogeneous artwork KG from a simple-wikidata-db staged directory.
# It validates required subfolders under the staged dir, runs data_dump.py to
# produce the graph JSONL files, then exports a .pt (PyTorch / PyG HeteroData).
#
# Usage:
#   bash /home/ghr/angel/gnn/build_kg_from_staged.sh \
#     --staged_dir /datasets/v2/c/angel/g/staged_en \
#     --out_dir /home/ghr/angel/gnn/graph_output \
#     --format pyg \
#     --limit 5000 \
#     --include_incoming
#
PY="${PYTHON:-python3}"
CODE_ROOT="${CODE_ROOT:-/home/ghr/angel/gnn}"
DATA_DUMP="${CODE_ROOT}/data_dump.py"
EXPORT_PT="${CODE_ROOT}/export_to_pt.py"

# Defaults (can be overridden by flags)
STAGED_DIR="/home/ghr/staged_en"
OUT_DIR="${CODE_ROOT}/graph_output"
GRAPH_DIR=""   # computed from OUT_DIR
PT_OUT=""      # computed from OUT_DIR
ROOT_ART_QID="${ROOT_ART_QID:-Q838948}"
LIMIT="${LIMIT:-5000}"
MAX_PASSES="${MAX_PASSES:-3}"
INCLUDE_INCOMING=1
FORMAT="${FORMAT:-pyg}"  # or "basic"

usage() {
  cat <<EOF
Build a heterogeneous artwork KG (.pt) from a simple-wikidata-db staged dir.

Staged data:
  --staged_dir PATH       Path to staged data (default: ${STAGED_DIR})

Optional:
  --out_dir PATH          Output root for graph files and .pt (default: ${OUT_DIR})
  --format {basic|pyg}    Export format for .pt (default: ${FORMAT})
  --root_art_qid QID      Root art class QID (default: ${ROOT_ART_QID})
  --limit N               Max artworks to seed (default: ${LIMIT})
  --max_passes N          Subclass-of passes (default: ${MAX_PASSES})
  --include_incoming      Include incoming entity_rels to artworks (default: on)
  --python PATH           Python interpreter (default: ${PY})

Examples:
  $(basename "$0") --out_dir ${OUT_DIR}
  $(basename "$0") --staged_dir /home/ghr/staged_en --format basic
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staged_dir) STAGED_DIR="${2:-}"; shift 2 ;;
    --out_dir) OUT_DIR="${2:-}"; shift 2 ;;
    --format) FORMAT="${2:-}"; shift 2 ;;
    --root_art_qid) ROOT_ART_QID="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --max_passes) MAX_PASSES="${2:-}"; shift 2 ;;
    --include_incoming) INCLUDE_INCOMING=1; shift 1 ;;
    --python) PY="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -d "${STAGED_DIR}" ]]; then
  echo "Error: staged_dir not found: ${STAGED_DIR}" >&2
  exit 1
fi

# Compute final paths
GRAPH_DIR="${OUT_DIR%/}/artwork_graph"
PT_OUT="${OUT_DIR%/}/graph.${FORMAT}.pt"

# Validate presence of expected subfolders (warn if missing)
echo "Checking staged tables under: ${STAGED_DIR}"
required=( "labels" "descriptions" "aliases" "entity_rels" "entity_values" "qualifiers" "wikipedia_links" )
missing_any=0
for t in "${required[@]}"; do
  if [[ ! -d "${STAGED_DIR}/${t}" ]]; then
    echo "WARN: Missing table: ${t}"
    missing_any=1
  fi
done
if [[ "${missing_any}" -eq 1 ]]; then
  echo "Proceeding despite missing tables; some edges/attributes may be absent."
fi

mkdir -p "${OUT_DIR}" "${GRAPH_DIR}"

echo "Running data_dump.py to build graph JSONL..."
set -x
${PY} "${DATA_DUMP}" \
  --db_dir "${STAGED_DIR}" \
  --root_art_qid "${ROOT_ART_QID}" \
  --limit "${LIMIT}" \
  --max_passes "${MAX_PASSES}" \
  --out "${OUT_DIR%/}/artworks_subset.jsonl" \
  --graph_out_dir "${GRAPH_DIR}" \
  $( (( INCLUDE_INCOMING == 1 )) && printf -- "--include_incoming_rels" )
set +x

echo "Exporting ${FORMAT} .pt to: ${PT_OUT}"
set -x
${PY} "${EXPORT_PT}" \
  --graph_dir "${GRAPH_DIR}" \
  --out "${PT_OUT}" \
  --format "${FORMAT}"
set +x

echo "Done."
echo "Staged dir: ${STAGED_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Graph dir: ${GRAPH_DIR}"
echo "PT file:   ${PT_OUT}"

