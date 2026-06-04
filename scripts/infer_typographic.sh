#!/usr/bin/env bash
set -euo pipefail

SUBMIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SUBMIT_ROOT/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-aqua_rat}"
INPUT_JSONL="${INPUT_JSONL:-$SUBMIT_ROOT/data/$DATASET/T-OR/output.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SUBMIT_ROOT/outputs/$DATASET/T-OR}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_DIR/infer_gpt5.1.jsonl}"
PROFILE="${PROFILE:-gpt5.1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "Error: input JSONL not found: $INPUT_JSONL"
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_JSONL")"

"$PYTHON_BIN" "$SUBMIT_ROOT/src/run.py" infer \
  --data "$INPUT_JSONL" \
  --output "$OUTPUT_JSONL" \
  --profile "$PROFILE" \
  --task-type img_reasoning \
  --max-tokens "$MAX_TOKENS"

echo "Typographic infer input:  $INPUT_JSONL"
echo "Typographic infer output: $OUTPUT_JSONL"
