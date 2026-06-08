#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_STR="${MODELS_STR:-gpt5.1 claude-sonnet-4.5 kimi-k2.5 gemini2.5 qwen3vl}"
BENCHMARKS_STR="${BENCHMARKS_STR:-aqua_rat gpqa gsm8k scienceqa_img zebra-cot}"
TASK_TYPES_STR="${TASK_TYPES_STR:-no_reasoning text_reasoning img_reasoning}"
TOKEN_RATIOS_STR="${TOKEN_RATIOS_STR:-0.2 0.4 0.6 0.8 1.0}"

START_INDEX="${START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/main_exp}"
PROFILES_CONFIG="${PROFILES_CONFIG:-src/configs/profiles.yaml}"
JUDGE_PROFILE="${JUDGE_PROFILE:-llmjudge}"
NO_LLM_JUDGE="${NO_LLM_JUDGE:-0}"
DRY_RUN="${DRY_RUN:-0}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

read -r -a MODELS <<< "$MODELS_STR"
read -r -a BENCHMARKS <<< "$BENCHMARKS_STR"
read -r -a TASK_TYPES <<< "$TASK_TYPES_STR"
read -r -a TOKEN_RATIOS <<< "$TOKEN_RATIOS_STR"

die() {
  echo "Error: $*" >&2
  exit 1
}

require_nonempty_array() {
  local name="$1"
  local size="$2"
  (( size > 0 )) || die "$name is empty"
}

validate_bool() {
  local name="$1"
  local value="$2"
  [[ "$value" == "0" || "$value" == "1" ]] || die "$name must be 0 or 1, got: $value"
}

validate_benchmark() {
  case "$1" in
    aqua_rat|gpqa|gsm8k|scienceqa_img|zebra-cot) ;;
    *) die "unsupported benchmark: $1" ;;
  esac
}

validate_task_type() {
  case "$1" in
    no_reasoning|text_reasoning|img_reasoning) ;;
    *) die "unsupported task type: $1" ;;
  esac
}

validate_ratio() {
  case "$1" in
    0.2|0.4|0.6|0.8|1.0) ;;
    *) die "unsupported token ratio: $1" ;;
  esac
}

baseline_jsonl() {
  case "$1" in
    gpqa) echo "data/gpqa/GPQA-diamond.jsonl" ;;
    scienceqa_img) echo "data/scienceqa_img/science_qa.jsonl" ;;
    *) echo "data/$1/$1.jsonl" ;;
  esac
}

image_jsonl() {
  local benchmark="$1"
  local ratio="$2"
  if [[ "$ratio" == "1.0" ]]; then
    echo "data/${benchmark}/T-OR/output.jsonl"
  else
    echo "data/${benchmark}/T-OR-${ratio}/output.jsonl"
  fi
}

setting_name() {
  local task_type="$1"
  local ratio="${2:-}"
  if [[ "$task_type" == "img_reasoning" ]]; then
    echo "img_reasoning_ratio_${ratio}"
  else
    echo "$task_type"
  fi
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

validate_profiles() {
  [[ -f "$PROFILES_CONFIG" ]] || die "profiles config not found: $PROFILES_CONFIG"
  local required_profiles=("${MODELS[@]}")
  if [[ "$NO_LLM_JUDGE" == "0" ]]; then
    required_profiles+=("$JUDGE_PROFILE")
  fi
  "$PYTHON_BIN" - "$PROFILES_CONFIG" "${required_profiles[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
models = sys.argv[2:]
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
configured = config.get("models") or {}
missing = [model for model in models if model not in configured]
if missing:
    raise SystemExit(f"Error: missing required profiles in {config_path}: {' '.join(missing)}")
PY
}

validate_inputs() {
  local benchmark task_type ratio input_jsonl
  for benchmark in "${BENCHMARKS[@]}"; do
    for task_type in "${TASK_TYPES[@]}"; do
      if [[ "$task_type" == "img_reasoning" ]]; then
        for ratio in "${TOKEN_RATIOS[@]}"; do
          input_jsonl="$(image_jsonl "$benchmark" "$ratio")"
          [[ -f "$input_jsonl" ]] || die "input JSONL not found: $input_jsonl"
        done
      else
        input_jsonl="$(baseline_jsonl "$benchmark")"
        [[ -f "$input_jsonl" ]] || die "input JSONL not found: $input_jsonl"
      fi
    done
  done
}

run_experiment() {
  local index="$1"
  local total="$2"
  local model="$3"
  local benchmark="$4"
  local task_type="$5"
  local input_jsonl="$6"
  local ratio="${7:-}"
  local setting output_dir output_jsonl
  local cmd

  setting="$(setting_name "$task_type" "$ratio")"
  output_dir="$OUTPUT_ROOT/$model/$benchmark/$setting"
  output_jsonl="$output_dir/output.jsonl"

  cmd=(
    "$PYTHON_BIN" src/run.py infer
    --data "$input_jsonl"
    --output "$output_jsonl"
    --profile "$model"
    --profiles-config "$PROFILES_CONFIG"
    --task-type "$task_type"
    --start-index "$START_INDEX"
    --judge-profile "$JUDGE_PROFILE"
  )

  [[ -z "$MAX_SAMPLES" ]] || cmd+=(--max-samples "$MAX_SAMPLES")
  [[ -z "$MAX_TOKENS" ]] || cmd+=(--max-tokens "$MAX_TOKENS")
  [[ "$NO_LLM_JUDGE" == "0" ]] || cmd+=(--no-llm-judge)

  echo "[$index/$total] model=$model benchmark=$benchmark setting=$setting"
  echo "  input:  $input_jsonl"
  echo "  output: $output_jsonl"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "${cmd[@]}"
  else
    mkdir -p "$output_dir"
    "${cmd[@]}"
  fi
}

require_nonempty_array MODELS_STR "${#MODELS[@]}"
require_nonempty_array BENCHMARKS_STR "${#BENCHMARKS[@]}"
require_nonempty_array TASK_TYPES_STR "${#TASK_TYPES[@]}"
validate_bool DRY_RUN "$DRY_RUN"
validate_bool VALIDATE_INPUTS "$VALIDATE_INPUTS"
validate_bool NO_LLM_JUDGE "$NO_LLM_JUDGE"
[[ "$START_INDEX" =~ ^[0-9]+$ ]] || die "START_INDEX must be a non-negative integer"
[[ -z "$MAX_SAMPLES" || "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "MAX_SAMPLES must be a positive integer"
[[ -z "$MAX_TOKENS" || "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS must be a positive integer"

for benchmark in "${BENCHMARKS[@]}"; do
  validate_benchmark "$benchmark"
done
for task_type in "${TASK_TYPES[@]}"; do
  validate_task_type "$task_type"
done
if [[ " ${TASK_TYPES[*]} " == *" img_reasoning "* ]]; then
  require_nonempty_array TOKEN_RATIOS_STR "${#TOKEN_RATIOS[@]}"
  for ratio in "${TOKEN_RATIOS[@]}"; do
    validate_ratio "$ratio"
  done
fi

validate_profiles
if [[ "$VALIDATE_INPUTS" == "1" ]]; then
  validate_inputs
fi

settings_per_pair=0
for task_type in "${TASK_TYPES[@]}"; do
  if [[ "$task_type" == "img_reasoning" ]]; then
    settings_per_pair=$((settings_per_pair + ${#TOKEN_RATIOS[@]}))
  else
    settings_per_pair=$((settings_per_pair + 1))
  fi
done
total_experiments=$((${#MODELS[@]} * ${#BENCHMARKS[@]} * settings_per_pair))

echo "Main experiment matrix"
echo "  models:       ${MODELS[*]}"
echo "  benchmarks:   ${BENCHMARKS[*]}"
echo "  task types:   ${TASK_TYPES[*]}"
echo "  token ratios: ${TOKEN_RATIOS[*]}"
echo "  experiments:  $total_experiments"
echo "  output root:  $OUTPUT_ROOT"
echo "  dry run:      $DRY_RUN"
echo

experiment_index=1
for model in "${MODELS[@]}"; do
  for benchmark in "${BENCHMARKS[@]}"; do
    for task_type in "${TASK_TYPES[@]}"; do
      if [[ "$task_type" == "img_reasoning" ]]; then
        for ratio in "${TOKEN_RATIOS[@]}"; do
          run_experiment \
            "$experiment_index" "$total_experiments" \
            "$model" "$benchmark" "$task_type" \
            "$(image_jsonl "$benchmark" "$ratio")" "$ratio"
          experiment_index=$((experiment_index + 1))
        done
      else
        run_experiment \
          "$experiment_index" "$total_experiments" \
          "$model" "$benchmark" "$task_type" \
          "$(baseline_jsonl "$benchmark")"
        experiment_index=$((experiment_index + 1))
      fi
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry-run completed: $total_experiments commands generated."
else
  echo "Main experiments completed: $total_experiments runs."
fi
echo "Results root: $OUTPUT_ROOT"
