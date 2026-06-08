#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-data}"
BENCHMARKS_STR="${BENCHMARKS_STR:-aqua_rat gpqa gsm8k scienceqa_img zebra-cot}"
TOKEN_RATIOS_STR="${TOKEN_RATIOS_STR:-0.2 0.4 0.6 0.8}"
WORKERS="${WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

read -r -a BENCHMARKS <<< "$BENCHMARKS_STR"
read -r -a TOKEN_RATIOS <<< "$TOKEN_RATIOS_STR"

die() {
  echo "Error: $*" >&2
  exit 1
}

[[ ${#BENCHMARKS[@]} -gt 0 ]] || die "BENCHMARKS_STR is empty"
[[ ${#TOKEN_RATIOS[@]} -gt 0 ]] || die "TOKEN_RATIOS_STR is empty"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "WORKERS must be a positive integer"
[[ -f src/utils/image_scaling.py ]] || die "missing scaler: src/utils/image_scaling.py"

for ratio in "${TOKEN_RATIOS[@]}"; do
  case "$ratio" in
    0.2|0.4|0.6|0.8) ;;
    *) die "unsupported token ratio: $ratio" ;;
  esac
done

temp_dirs=()
cleanup() {
  local temp_dir
  for temp_dir in "${temp_dirs[@]}"; do
    [[ ! -e "$temp_dir" ]] || rm -rf "$temp_dir"
  done
}
trap cleanup EXIT

# Validate the complete matrix before creating any output.
for benchmark in "${BENCHMARKS[@]}"; do
  source_dir="$DATA_ROOT/$benchmark/T-OR"
  [[ -f "$source_dir/output.jsonl" ]] || die "missing source JSONL: $source_dir/output.jsonl"
  [[ -d "$source_dir/images" ]] || die "missing source image directory: $source_dir/images"
  for ratio in "${TOKEN_RATIOS[@]}"; do
    target_dir="$DATA_ROOT/$benchmark/T-OR-$ratio"
    [[ ! -e "$target_dir" ]] || die "target already exists: $target_dir"
  done
done

total=$((${#BENCHMARKS[@]} * ${#TOKEN_RATIOS[@]}))
index=1

for benchmark in "${BENCHMARKS[@]}"; do
  source_dir="$DATA_ROOT/$benchmark/T-OR"
  for ratio in "${TOKEN_RATIOS[@]}"; do
    target_dir="$DATA_ROOT/$benchmark/T-OR-$ratio"
    temp_dir="$DATA_ROOT/$benchmark/.T-OR-$ratio.tmp.$$"
    temp_dirs+=("$temp_dir")

    echo "[$index/$total] benchmark=$benchmark ratio=$ratio"
    "$PYTHON_BIN" src/utils/image_scaling.py \
      "$source_dir/images" \
      "$temp_dir/images" \
      --scale "$ratio" \
      --workers "$WORKERS"

    "$PYTHON_BIN" - \
      "$source_dir/output.jsonl" \
      "$source_dir/images" \
      "$temp_dir/output.jsonl" \
      "$temp_dir/images" \
      "$ratio" <<'PY'
import json
import math
import sys
from pathlib import Path

from PIL import Image

source_jsonl = Path(sys.argv[1])
source_images = Path(sys.argv[2])
output_jsonl = Path(sys.argv[3])
output_images = Path(sys.argv[4])
scale = float(sys.argv[5])

source_files = sorted(source_images.rglob("*.png"))
relative_by_name: dict[str, Path] = {}
for source_file in source_files:
    name = source_file.name
    if name in relative_by_name:
        raise SystemExit(f"Duplicate source image basename: {name}")
    relative_by_name[name] = source_file.relative_to(source_images)


def normalize_image_path(value: str) -> str:
    name = Path(value).name
    relative = relative_by_name.get(name)
    if relative is None:
        raise SystemExit(f"Referenced image not found under {source_images}: {value}")
    return (Path("images") / relative).as_posix()


rows = []
with source_jsonl.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("image_path")
        if isinstance(value, list):
            row["image_path"] = [normalize_image_path(str(item)) for item in value]
        elif isinstance(value, str) and value.strip():
            row["image_path"] = normalize_image_path(value)
        else:
            raise SystemExit(f"Missing or invalid image_path at line {line_number}")
        rows.append(row)

output_jsonl.parent.mkdir(parents=True, exist_ok=True)
with output_jsonl.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

output_files = sorted(output_images.rglob("*.png"))
if len(output_files) != len(source_files):
    raise SystemExit(
        f"Image count mismatch: source={len(source_files)} output={len(output_files)}"
    )

for source_file in source_files:
    relative = source_file.relative_to(source_images)
    output_file = output_images / relative
    if not output_file.is_file():
        raise SystemExit(f"Missing scaled image: {output_file}")
    with Image.open(source_file) as source, Image.open(output_file) as output:
        expected = (
            max(1, round(source.width * math.sqrt(scale))),
            max(1, round(source.height * math.sqrt(scale))),
        )
        if output.size != expected:
            raise SystemExit(
                f"Unexpected size for {output_file}: got={output.size} expected={expected}"
            )

print(f"Wrote {len(rows)} rows and validated {len(output_files)} images")
PY

    index=$((index + 1))
  done
done

# Publish only after every temporary dataset has passed validation.
for benchmark in "${BENCHMARKS[@]}"; do
  for ratio in "${TOKEN_RATIOS[@]}"; do
    temp_dir="$DATA_ROOT/$benchmark/.T-OR-$ratio.tmp.$$"
    target_dir="$DATA_ROOT/$benchmark/T-OR-$ratio"
    [[ ! -e "$target_dir" ]] || die "target appeared during generation: $target_dir"
    mv "$temp_dir" "$target_dir"
  done
done

temp_dirs=()
echo "Generated $total T-OR ratio datasets under $DATA_ROOT."
