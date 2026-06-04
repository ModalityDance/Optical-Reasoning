#!/usr/bin/env bash
set -euo pipefail

SUBMIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SUBMIT_ROOT:${PYTHONPATH:-}"
cd "$SUBMIT_ROOT"

DATASET="${DATASET:-aqua_rat}"
DATASET_DIR="${DATASET_DIR:-$SUBMIT_ROOT/data/$DATASET}"
INPUT_JSONL="${INPUT_JSONL:-$DATASET_DIR/aqua_rat.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-$DATASET_DIR/G-OR}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_BASE/output.jsonl}"
PROFILE="${PROFILE:-nano-banana-pro}"
SIZE="${SIZE:-2:3}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-b64_json}"
ASPECT_RATIO="${ASPECT_RATIO:-}"
TOKEN_SIZING_MODEL="${TOKEN_SIZING_MODEL:-qwen3-vl}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_BASE" "$(dirname "$OUTPUT_JSONL")"

"$PYTHON_BIN" - \
  "$INPUT_JSONL" \
  "$OUTPUT_JSONL" \
  "$OUTPUT_BASE" \
  "$PROFILE" \
  "$SIZE" \
  "$RESPONSE_FORMAT" \
  "$ASPECT_RATIO" \
  "$TOKEN_SIZING_MODEL" <<'PY'
import sys

from src.render.graphical_render import generate_t2i_image_dataset

(
    input_jsonl,
    output_jsonl,
    output_base,
    profile,
    size,
    response_format,
    aspect_ratio,
    token_sizing_model,
) = sys.argv[1:]

stats = generate_t2i_image_dataset(
    input_jsonl=input_jsonl,
    output_jsonl=output_jsonl,
    output_base=output_base,
    profile_name=profile,
    size=size,
    response_format=response_format,
    aspect_ratio=aspect_ratio or None,
    token_sizing_model=token_sizing_model,
)

print(f"Stats: {stats}")
PY

echo "Graphical render output: $OUTPUT_BASE/images"
echo "Graphical JSONL: $OUTPUT_JSONL"
