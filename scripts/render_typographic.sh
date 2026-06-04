#!/usr/bin/env bash
set -euo pipefail

SUBMIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SUBMIT_ROOT/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-aqua_rat}"
DATASET_DIR="${DATASET_DIR:-$SUBMIT_ROOT/data/$DATASET}"
INPUT_JSONL="${INPUT_JSONL:-$DATASET_DIR/aqua_rat.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_DIR/T-OR}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_DIR/output.jsonl}"
TOKEN_MODEL="${TOKEN_MODEL:-qwen3-vl}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_JSONL")"

"$PYTHON_BIN" "$SUBMIT_ROOT/src/run.py" render-jsonl \
  --data "$INPUT_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --output-jsonl "$OUTPUT_JSONL" \
  --text-field solution \
  --image-field image_path \
  --token-model "$TOKEN_MODEL"

echo "Typographic render output: $OUTPUT_DIR"
echo "Typographic JSONL: $OUTPUT_JSONL"
