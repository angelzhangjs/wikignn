#!/usr/bin/env bash
set -euo pipefail

# Configuration
LANGUAGE_ID="en"
ROOT_ART_QID="Q838948"          # work of art
LIMIT=100                      # number of artworks to seed the graph
MAX_PASSES=2                    # subclass-of passes
INCLUDE_INCOMING_RELS=1         # 1 = include incoming Q->Q edges; 0 = only outgoing

# Code root (where Python scripts live) and Data root (where data is stored)
CODE_ROOT="${CODE_ROOT:-/home/ghr/angel/gnn}"
DATA_ROOT="${DATA_ROOT:-/datasets/v2p/current/angel/gnn}"

# Ensure local modules are importable
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

# Paths
# Base output directory for generated graph artifacts (override with OUTPUT_DIR if desired)
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_ROOT}/graph_output}"

DUMP_FILE="${DATA_ROOT}/data/latest-all.json.gz"
STAGED_DIR="${STAGED_DIR:-${DATA_ROOT}/staged_${LANGUAGE_ID}}"
GRAPH_DIR="${GRAPH_DIR:-${OUTPUT_DIR}/artwork_graph}"
ARTWORKS_LIST="${ARTWORKS_LIST:-${OUTPUT_DIR}/artworks_subset.jsonl}"
# PyG .pt export target (override with PT_OUT env if desired)
PT_OUT="${PT_OUT:-${OUTPUT_DIR}/graph.pt}"

DATA_DUMP="${CODE_ROOT}/data_dump.py"
PREPROCESS="${CODE_ROOT}/preprocess_dump.py"
VIS_SCRIPT="${CODE_ROOT}/graph_visualization.py"
EXPORT_PT="${CODE_ROOT}/export_to_pt.py"

# Optional: limit lines read for a quick sample (set to -1 for full run)
NUM_LINES_READ="-1"   # e.g., "2000000" for a quicker partial stage

download_dump() {
  mkdir -p "$(dirname "$DUMP_FILE")"
  if [[ -f "$DUMP_FILE" ]]; then
    echo "Dump already exists at $DUMP_FILE"
    return
  fi
  echo "Downloading Wikidata dump..."
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --max-connection-per-server 16 -o "$(basename "$DUMP_FILE")" \
      "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz" \
      -d "$(dirname "$DUMP_FILE")"
  else
    wget -O "$DUMP_FILE" "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz"
  fi
}

stage_dump() {
  mkdir -p "$STAGED_DIR"
  echo "Staging to $STAGED_DIR (language=${LANGUAGE_ID})"
  if [[ "${NUM_LINES_READ}" != "-1" ]]; then
    python3 "$PREPROCESS" \
      --input_file "$DUMP_FILE" \
      --out_dir "$STAGED_DIR" \
      --language_id "$LANGUAGE_ID" \
      --num_lines_read "$NUM_LINES_READ"
  else
    python3 "$PREPROCESS" \
      --input_file "$DUMP_FILE" \
      --out_dir "$STAGED_DIR" \
      --language_id "$LANGUAGE_ID"
  fi
}

dump_artwork_graph() {
  mkdir -p "$GRAPH_DIR"
  echo "Building artwork graph in $GRAPH_DIR"
  extra_flag=()
  if [[ "$INCLUDE_INCOMING_RELS" == "1" ]]; then
    extra_flag+=(--include_incoming_rels)
  fi
  python3 "$DATA_DUMP" \
    --db_dir "$STAGED_DIR" \
    --root_art_qid "$ROOT_ART_QID" \
    --limit "$LIMIT" \
    --max_passes "$MAX_PASSES" \
    --out "$ARTWORKS_LIST" \
    --graph_out_dir "$GRAPH_DIR" \
    "${extra_flag[@]}"
}

export_graph_pt() {
  echo "Exporting PyG .pt to $PT_OUT"
  mkdir -p "$(dirname "$PT_OUT")"
  python3 "$EXPORT_PT" \
    --graph_dir "$GRAPH_DIR" \
    --out "$PT_OUT" \
    --format pyg
}

visualize_optional() {
  # Optional: visualize using graph_visualization.py (requires matplotlib)
  echo "Visualizing graph (optional)..."
  GRAPH_OUT_DIR="$GRAPH_DIR" python3 "$VIS_SCRIPT" || true
}

main() {
  if [[ ! -d "$STAGED_DIR/entity_rels" ]]; then
    echo "No staged data at $STAGED_DIR; downloading and staging..."
    download_dump
    stage_dump
  else
    echo "Staged directory exists at $STAGED_DIR; skipping download/staging."
  fi
  mkdir -p "$OUTPUT_DIR"
  dump_artwork_graph
  export_graph_pt
  # visualize_optional   # uncomment to visualize immediately
  echo "Done."
  echo "Data root: $DATA_ROOT"
  echo "Graph dir: $GRAPH_DIR"
  echo "Output dir: $OUTPUT_DIR"
  echo "PT out: $PT_OUT"
  echo "Artworks list: $ARTWORKS_LIST"
}

main "$@"


