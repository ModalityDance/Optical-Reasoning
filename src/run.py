"""Minimal CLI for LLM inference and reasoning-text rendering."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import Predictor
from src.inference.evaluation import DEFAULT_CONFIG_PATH, DEFAULT_PROFILE, evaluate_results, load_model_config
from src.render.typographic_render import render_dense_latex_to_png


REASONING_IMAGE_PREFIX = "reasoning_image_"
TASK_TYPES = ("no_reasoning", "free_reasoning", "text_reasoning", "img_reasoning")
DEFAULT_INFER_PROFILE = "default"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(raw_path: Any, data_path: str | Path) -> str | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None

    path = Path(text)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())

    data_dir = Path(data_path).resolve().parent
    for candidate_dir in (data_dir, *data_dir.parents):
        candidate = candidate_dir / path
        if candidate.exists():
            return str(candidate.resolve())
    return str((data_dir / path).resolve())


def resolve_paths(raw_paths: Any, data_path: str | Path) -> list[str]:
    if raw_paths is None:
        return []
    if isinstance(raw_paths, list):
        return [
            resolved
            for raw_path in raw_paths
            if (resolved := resolve_path(raw_path, data_path))
        ]
    resolved = resolve_path(raw_paths, data_path)
    return [resolved] if resolved else []


def problem_text(entry: dict[str, Any]) -> str:
    return str(entry.get("problem") or entry.get("question") or "")


def build_prompt(entry: dict[str, Any], task_type: str) -> str:
    problem = problem_text(entry)
    if task_type == "text_reasoning":
        solution = str(entry.get("solution") or "").strip()
        return f"{problem}\n\n{solution}" if solution else problem
    return problem


def reasoning_image_sort_key(field_name: str) -> tuple[int, str]:
    suffix = field_name.removeprefix(REASONING_IMAGE_PREFIX)
    if suffix.isdigit():
        return int(suffix), ""
    return 10**9, field_name


def collect_reasoning_images(entry: dict[str, Any], data_path: str | Path) -> list[str]:
    field_names = sorted(
        (key for key in entry if key.startswith(REASONING_IMAGE_PREFIX)),
        key=reasoning_image_sort_key,
    )
    images = [
        resolved
        for field_name in field_names
        if (resolved := resolve_path(entry.get(field_name), data_path))
    ]
    if images:
        return images
    return resolve_paths(entry.get("image_path"), data_path)


def build_input_data(entry: dict[str, Any], task_type: str, data_path: str | Path) -> dict[str, Any]:
    input_data: dict[str, Any] = {}
    question_image = resolve_path(entry.get("question_image"), data_path)
    if question_image:
        input_data["question_image"] = question_image

    if task_type == "img_reasoning":
        images = collect_reasoning_images(entry, data_path)
        if images:
            input_data["image_path"] = images
    return input_data


def extract_boxed_content(text: str) -> str | None:
    for marker in (r"\boxed", "\\\\boxed"):
        start = text.find(marker)
        if start < 0:
            continue
        brace_start = text.find("{", start + len(marker))
        if brace_start < 0:
            continue
        depth = 0
        for index in range(brace_start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1:index].strip()

    match = re.search(r"\\boxed\s*\(([^)]*)\)", text)
    return match.group(1).strip() if match else None


def parse_prediction(prediction: str) -> str:
    boxed = extract_boxed_content(prediction.strip())
    return boxed if boxed is not None else prediction.strip()


def setup_logging(verbose: bool = False) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG" if verbose else "INFO",
        colorize=True,
    )


def run_infer(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)

    profile_config = load_model_config(args.profile, args.profiles_config)
    if profile_config is None:
        if not (args.model and args.base_url and (args.api_key or os.environ.get(args.api_key_env))):
            raise ValueError(
                f"Missing inference profile {args.profile!r}. "
                f"Check --profiles-config or pass --model/--base-url/--api-key."
            )
    api_key = args.api_key or os.environ.get(args.api_key_env) or (
        profile_config.api_key if profile_config else None
    )
    base_url = args.base_url or (profile_config.base_url if profile_config else None)
    model = args.model or (profile_config.model if profile_config else None)
    temperature = (
        args.temperature
        if args.temperature is not None
        else profile_config.temperature if profile_config is not None else 0.0
    )
    reasoning_effort = (
        args.reasoning_effort
        if args.reasoning_effort is not None
        else profile_config.reasoning_effort if profile_config is not None else None
    )

    rows = read_jsonl(args.data)
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    end_index = None if args.max_samples is None else args.start_index + args.max_samples
    selected_rows = rows[args.start_index:end_index]

    predictor = Predictor(
        model=model,
        base_url=base_url,
        api_key=api_key,
        task_type=args.task_type,
        max_tokens=args.max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )

    results: list[dict[str, Any]] = []
    for local_index, entry in enumerate(selected_rows):
        source_index = args.start_index + local_index
        result = dict(entry)
        result["source_index"] = source_index
        try:
            prompt = build_prompt(entry, args.task_type)
            input_data = build_input_data(entry, args.task_type, args.data)
            prediction_result = predictor.predict(input_data=input_data, prompt=prompt)
            prediction = str(prediction_result.get("prediction") or "")
            result.update(
                {
                    "prediction": prediction,
                    "parsed_prediction": parse_prediction(prediction),
                    "input_tokens": int(prediction_result.get("input_tokens") or 0),
                    "output_tokens": int(prediction_result.get("output_tokens") or 0),
                    "success": True,
                }
            )
        except Exception as exc:
            result.update({"success": False, "error": str(exc)})
            logger.error(f"sample {source_index} failed: {exc}")
        results.append(result)

        if args.save_interval and len(results) % args.save_interval == 0:
            write_jsonl(args.output, results)

    write_jsonl(args.output, results)
    success_count = sum(1 for item in results if item.get("success"))
    logger.info(f"Wrote {len(results)} rows to {args.output}; success={success_count}")
    if args.evaluate:
        evaluate_results(
            args.output,
            metrics_path=args.metrics_output,
            dataset_tag=args.dataset_tag,
            judge_profile=args.judge_profile,
            profiles_config=args.profiles_config,
            enable_llm_judge=not args.no_llm_judge,
        )
    return 0


def run_render_jsonl(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    rows = read_jsonl(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl_dir = Path(args.output_jsonl).parent.resolve() if args.output_jsonl else output_dir.resolve()

    rendered_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(rows):
        if args.max_samples is not None and index >= args.max_samples:
            break
        text = str(entry.get(args.text_field) or "")
        entry_id = str(entry.get("id") or entry.get("uid") or index)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", entry_id).strip("._") or str(index)
        image_path = output_dir / f"{safe_id}.png"
        if "reasoning_token" not in entry:
            raise ValueError(f"Entry {entry_id} is missing reasoning_token")
        final_size = render_dense_latex_to_png(
            text=text,
            output_path=str(image_path),
            reasoning_token=int(entry["reasoning_token"]),
            model_name=args.token_model,
        )
        rendered = dict(entry)
        rendered[args.image_field] = os.path.relpath(image_path.resolve(), output_jsonl_dir)
        rendered_rows.append(rendered)
        logger.info(f"rendered {entry_id} -> {image_path} ({final_size[0]}x{final_size[1]})")

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rendered_rows)
        logger.info(f"Wrote rendered JSONL to {args.output_jsonl}")
    return 0


def add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", "--data-path", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--profile", default=DEFAULT_INFER_PROFILE)
    parser.add_argument("--model", "--model-name", default=None, help="Override profile model")
    parser.add_argument("--base-url", default=None, help="Override profile API base URL")
    parser.add_argument("--api-key", default=None, help="Override profile API key")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable used when --api-key is omitted",
    )
    parser.add_argument("--task-type", choices=TASK_TYPES, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-output", default=None, help="Defaults to metrics.json beside --output")
    parser.add_argument("--dataset-tag", default=None, help="Optional dataset tag for evaluation routing")
    parser.add_argument("--judge-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--profiles-config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--no-llm-judge", action="store_true", help="Run static evaluation only")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    infer_parser = subparsers.add_parser("infer", help="Run LLM inference on a JSONL dataset")
    add_infer_args(infer_parser)
    infer_parser.set_defaults(func=run_infer)

    render_parser = subparsers.add_parser("render-jsonl", help="Render a JSONL text field to PNG files")
    render_parser.add_argument("--data", required=True, help="Input JSONL path")
    render_parser.add_argument("--output-dir", required=True, help="Directory for rendered PNGs")
    render_parser.add_argument("--output-jsonl", default=None, help="Optional JSONL with image paths")
    render_parser.add_argument("--text-field", default="solution")
    render_parser.add_argument("--image-field", default="image_path")
    render_parser.add_argument("--token-model", default="qwen3-vl")
    render_parser.add_argument("--max-samples", type=int, default=None)
    render_parser.add_argument("--verbose", action="store_true")
    render_parser.set_defaults(func=run_render_jsonl)

    return parser


def normalize_legacy_args(argv: list[str]) -> list[str]:
    if not argv or argv[0] in {"infer", "render-jsonl", "-h", "--help"}:
        return argv
    legacy = list(argv)
    if "--exp-name" in legacy or "--experiment-name" in legacy:
        for flag in ("--exp-name", "--experiment-name"):
            if flag in legacy:
                index = legacy.index(flag)
                exp_name = legacy[index + 1]
                del legacy[index:index + 2]
                if "--output" not in legacy:
                    legacy.extend(["--output", str(Path("outputs") / exp_name / "results.jsonl")])
                break
    return ["infer", *legacy]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = normalize_legacy_args(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
