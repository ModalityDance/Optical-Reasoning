from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from openai import OpenAI

try:
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except Exception:  # pragma: no cover - optional dependency in submit environments
    parse = None
    verify = None
    ExprExtractionConfig = None
    LatexExtractionConfig = None


OPTION_LETTERS = tuple("ABCDE")
DEFAULT_PROFILE = "llmjudge"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "profiles.yaml"


@dataclass
class ModelConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float | None = None
    reasoning_effort: str | None = None


JudgeConfig = ModelConfig


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


def load_model_config(
    profile: str = DEFAULT_PROFILE,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ModelConfig | None:
    """Load an OpenAI-compatible model profile, with env overrides."""
    path = Path(config_path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile_data = dict(data["models"][profile])
    except Exception as exc:
        logger.warning(f"Cannot load profile {profile!r} from {path}: {exc}")
        return None

    env_prefix = profile.upper().replace("-", "_").replace(".", "_")
    api_key = os.getenv(f"{env_prefix}_API_KEY") or profile_data.get("api_key")
    base_url = os.getenv(f"{env_prefix}_BASE_URL") or profile_data.get("base_url")
    model = os.getenv(f"{env_prefix}_MODEL") or profile_data.get("model")
    if not (api_key and base_url and model):
        logger.warning(f"Profile {profile!r} is missing api_key/base_url/model")
        return None
    temperature = profile_data.get("temperature")
    return ModelConfig(
        model=str(model),
        base_url=str(base_url),
        api_key=str(api_key),
        temperature=float(temperature) if temperature is not None else None,
        reasoning_effort=profile_data.get("reasoning_effort"),
    )


def load_judge_config(
    profile: str = DEFAULT_PROFILE,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> JudgeConfig | None:
    """Load judge profile, with LLM_JUDGE_* env overrides as a final fallback."""
    config = load_model_config(profile=profile, config_path=config_path)
    if config is None:
        return None
    return JudgeConfig(
        model=os.getenv("LLM_JUDGE_MODEL") or config.model,
        base_url=os.getenv("LLM_JUDGE_BASE_URL") or config.base_url,
        api_key=os.getenv("LLM_JUDGE_API_KEY") or config.api_key,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
    )


def normalize_answer(text: Any) -> str:
    value = "" if text is None else str(text)
    return re.sub(r"\s+", " ", value.strip()).lower()


def extract_boxed_content(text: Any) -> str | None:
    value = "" if text is None else str(text)
    found: str | None = None
    index = value.find(r"\boxed")
    while index >= 0:
        cursor = index + len(r"\boxed")
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor < len(value) and value[cursor] == "{":
            depth = 0
            start = cursor + 1
            for end in range(cursor, len(value)):
                if value[end] == "{":
                    depth += 1
                elif value[end] == "}":
                    depth -= 1
                    if depth == 0:
                        found = value[start:end].strip()
                        break
        elif cursor < len(value) and value[cursor] == "(":
            end = value.find(")", cursor + 1)
            if end >= 0:
                found = value[cursor + 1:end].strip()
        index = value.find(r"\boxed", index + len(r"\boxed"))
    return found


def answer_candidates(text: Any) -> list[str]:
    raw = "" if text is None else str(text).strip()
    boxed = extract_boxed_content(raw)
    ordered = [boxed, raw] if boxed else [raw]
    candidates: list[str] = []
    for candidate in ordered:
        if not candidate:
            continue
        stripped = candidate.strip().strip("`")
        for opener, closer in (("$", "$"), ("(", ")"), ("[", "]"), ("{", "}")):
            if stripped.startswith(opener) and stripped.endswith(closer):
                stripped = stripped[1:-1].strip()
        if stripped and stripped not in candidates:
            candidates.append(stripped)
    return candidates


def extract_option_letter(text: Any, problem_text: str = "") -> str | None:
    letter_pattern = "".join(OPTION_LETTERS)
    for candidate in answer_candidates(text):
        match = re.fullmatch(
            rf"\s*[\(\[\{{]?\s*([{letter_pattern}{letter_pattern.lower()}])\s*[\)\]\}}]?\s*[.)]?\s*",
            candidate,
        )
        if match:
            return match.group(1).upper()
        match = re.search(
            rf"(?i)\b(?:answer|final answer|option|choice)\b[^A-Z]{{0,12}}([{letter_pattern}{letter_pattern.lower()}])\b",
            candidate,
        )
        if match:
            return match.group(1).upper()
        tokens = re.findall(rf"\b([{letter_pattern}{letter_pattern.lower()}])\b", candidate)
        if len(tokens) == 1 and len(candidate) <= 16:
            return tokens[0].upper()

    options = parse_options(problem_text)
    if not options:
        return None
    normalized_options = {letter: normalize_for_option_match(value) for letter, value in options.items()}
    for candidate in answer_candidates(text):
        normalized = normalize_for_option_match(candidate)
        matches = [
            letter
            for letter, option_text in normalized_options.items()
            if option_text and option_text == normalized
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def parse_options(problem_text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    pattern = re.compile(r"([A-E])(?:[.)])\s*(.*?)(?=(?:\n\s*[A-E][.)]\s*)|\Z)", re.S)
    text = problem_text.split("Options:", 1)[1] if "Options:" in problem_text else problem_text
    for match in pattern.finditer(text):
        value = match.group(2).strip()
        if value:
            options[match.group(1).upper()] = value
    return options


def normalize_for_option_match(text: Any) -> str:
    value = "" if text is None else str(text)
    value = extract_boxed_content(value) or value
    value = value.replace("$", " ")
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = re.sub(r"[^A-Za-z0-9.+\\/-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def ensure_boxed(text: str) -> str:
    stripped = text.strip()
    if r"\boxed{" in stripped:
        return stripped
    return rf"\boxed{{{stripped}}}"


def math_equal(prediction: str, answer: str) -> bool:
    if not prediction or not answer:
        return False
    if normalize_answer(prediction) == normalize_answer(answer):
        return True
    if parse is None or verify is None:
        return False
    try:
        extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        gold = parse(ensure_boxed(answer), extraction_config=extraction_config)
        pred = parse(ensure_boxed(prediction), extraction_config=extraction_config)
        return bool(verify(gold, pred))
    except Exception:
        return False


def infer_dataset_tag(path_like: str | Path | None) -> str | None:
    if path_like is None:
        return None
    text = str(path_like).replace("\\", "/").lower()
    for tag in ("aqua_rat", "gpqa", "scienceqa", "zebra_cot", "gsm8k"):
        if tag in text:
            return tag
    return None


def problem_text(entry: dict[str, Any]) -> str:
    return str(entry.get("problem") or entry.get("question") or "")


def prediction_text(entry: dict[str, Any]) -> str:
    return str(entry.get("parsed_prediction", entry.get("parsed_result", entry.get("prediction", ""))) or "")


def is_completed(entry: dict[str, Any]) -> bool:
    if entry.get("success") is False:
        return False
    if entry.get("fetch_success") is False or entry.get("process_success") is False:
        return False
    return True


def static_evaluate(entry: dict[str, Any], dataset_tag: str | None) -> dict[str, Any]:
    prediction = prediction_text(entry)
    answer = str(entry.get("answer", "") or "")
    problem = problem_text(entry)
    gold_label = extract_option_letter(answer, problem)
    pred_label = extract_option_letter(prediction, problem)
    is_choice = gold_label is not None or dataset_tag in {"aqua_rat", "gpqa", "scienceqa"}

    if not is_completed(entry):
        return {
            "correct": False,
            "eval_mode": "choice" if is_choice else "open_ended",
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "resolved_by": "inference_error",
        }

    if gold_label is not None:
        return {
            "correct": pred_label == gold_label,
            "eval_mode": "choice",
            "gold_label": gold_label,
            "predicted_label": pred_label,
            "resolved_by": "static_choice" if pred_label is not None else "unparsed_choice",
        }

    return {
        "correct": math_equal(prediction, answer),
        "eval_mode": "open_ended",
        "gold_label": None,
        "predicted_label": None,
        "resolved_by": "static_math",
    }


MULTIPLE_CHOICE_SYSTEM_PROMPT = """You judge whether a model's multiple-choice answer matches the gold label.
Return strict JSON only: {"verdict":"CORRECT"|"INCORRECT","predicted_label":"A"|"B"|"C"|"D"|"E"|"UNKNOWN"}."""

OPEN_ENDED_SYSTEM_PROMPT = """You judge whether a model's answer is mathematically equivalent to the gold answer.
Return strict JSON only: {"verdict":"CORRECT"|"INCORRECT"}."""


def build_choice_judge_prompt(entry: dict[str, Any], prediction: str, gold_label: str) -> str:
    return (
        f"Problem:\n{problem_text(entry)}\n\n"
        f"Gold label: {gold_label}\n\n"
        f"Model answer:\n{prediction}\n"
    )


def build_open_ended_judge_prompt(entry: dict[str, Any], prediction: str, answer: str) -> str:
    return (
        f"Problem:\n{problem_text(entry)}\n\n"
        f"Gold answer:\n{answer}\n\n"
        f"Model answer:\n{prediction}\n"
    )


def parse_judge_response(raw_response: str) -> tuple[bool | None, str | None]:
    text = raw_response.strip()
    if not text:
        return None, None
    try:
        payload = json.loads(text)
        verdict_text = str(payload.get("verdict", "")).upper()
        predicted = extract_option_letter(payload.get("predicted_label") or payload.get("predicted_letter"))
        if verdict_text == "CORRECT":
            return True, predicted
        if verdict_text == "INCORRECT":
            return False, predicted
    except Exception:
        pass
    upper = text.upper()
    if "INCORRECT" in upper:
        return False, extract_option_letter(text)
    if re.search(r"\bCORRECT\b", upper):
        return True, extract_option_letter(text)
    return None, extract_option_letter(text)


class LLMJudge:
    def __init__(self, config: JudgeConfig):
        self.model = config.model
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def judge_choice(self, entry: dict[str, Any], prediction: str, gold_label: str) -> tuple[bool | None, str | None, str]:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=96,
            messages=[
                {"role": "system", "content": MULTIPLE_CHOICE_SYSTEM_PROMPT},
                {"role": "user", "content": build_choice_judge_prompt(entry, prediction, gold_label)},
            ],
        )
        raw = completion.choices[0].message.content or ""
        verdict, predicted_label = parse_judge_response(raw)
        return verdict, predicted_label, raw

    def judge_open_ended(self, entry: dict[str, Any], prediction: str, answer: str) -> tuple[bool | None, str]:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=96,
            messages=[
                {"role": "system", "content": OPEN_ENDED_SYSTEM_PROMPT},
                {"role": "user", "content": build_open_ended_judge_prompt(entry, prediction, answer)},
            ],
        )
        raw = completion.choices[0].message.content or ""
        verdict, _ = parse_judge_response(raw)
        return verdict, raw


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = record.get("index")
            if isinstance(index, int):
                records[index] = record
    return records


def write_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_llm_judge(
    entry: dict[str, Any],
    detail: dict[str, Any],
    judge: LLMJudge,
) -> tuple[bool | None, str | None, str]:
    prediction = prediction_text(entry)
    answer = str(entry.get("answer", "") or "")
    if detail["eval_mode"] == "choice" and detail.get("gold_label"):
        return judge.judge_choice(entry, prediction, str(detail["gold_label"]))
    verdict, raw = judge.judge_open_ended(entry, prediction, answer)
    return verdict, None, raw


def should_use_llm_judge(entry: dict[str, Any], detail: dict[str, Any]) -> bool:
    """Use LLM judge only as a fallback after static matching fails."""
    return is_completed(entry) and not bool(detail["correct"])


def evaluate_results(
    results_path: str | Path,
    metrics_path: str | Path | None = None,
    *,
    dataset_tag: str | None = None,
    judge_profile: str = DEFAULT_PROFILE,
    profiles_config: str | Path = DEFAULT_CONFIG_PATH,
    enable_llm_judge: bool = True,
) -> dict[str, Any]:
    results_file = Path(results_path)
    output_path = Path(metrics_path) if metrics_path else results_file.with_name("metrics.json")
    rows = read_jsonl(results_file)
    inferred_tag = dataset_tag or infer_dataset_tag(results_file)
    checkpoint_path = results_file.with_name(f"{results_file.stem}.llm_judge_checkpoint.jsonl")
    checkpoint_records = load_checkpoint(checkpoint_path)

    judge: LLMJudge | None = None
    if enable_llm_judge:
        config = load_judge_config(profile=judge_profile, config_path=profiles_config)
        if config is not None:
            try:
                judge = LLMJudge(config)
            except Exception as exc:
                logger.warning(f"LLM judge disabled: initialization failed: {exc}")

    details: list[dict[str, Any]] = []
    attempted = 0
    resolved = 0
    for index, entry in enumerate(rows):
        static = static_evaluate(entry, inferred_tag)
        detail = {
            "id": entry.get("id", entry.get("source_index", index)),
            "correct": bool(static["correct"]),
            "static_correct": bool(static["correct"]),
            "prediction": prediction_text(entry)[:500],
            "answer": str(entry.get("answer", "") or "")[:200],
            "input_tokens": int(entry.get("input_tokens") or 0),
            "output_tokens": int(entry.get("output_tokens") or 0),
            "eval_mode": static["eval_mode"],
            "gold_label": static["gold_label"],
            "predicted_label": static["predicted_label"],
            "resolved_by": static["resolved_by"],
            "llm_judge_used": False,
            "llm_judge_raw": "",
        }

        if judge is not None and should_use_llm_judge(entry, detail):
            record = checkpoint_records.get(index)
            if record is None:
                try:
                    verdict, predicted_label, raw = run_llm_judge(entry, detail, judge)
                except Exception as exc:
                    verdict, predicted_label, raw = None, None, f"LLM_JUDGE_ERROR: {exc}"
                record = {
                    "index": index,
                    "entry_id": detail["id"],
                    "verdict": verdict,
                    "predicted_label": predicted_label,
                    "raw_response": raw,
                }
                write_checkpoint(checkpoint_path, record)

            attempted += 1
            detail["llm_judge_used"] = True
            detail["llm_judge_raw"] = str(record.get("raw_response", "") or "")[:200]
            if record.get("predicted_label") is not None:
                detail["predicted_label"] = record["predicted_label"]
            if record.get("verdict") is not None:
                detail["correct"] = bool(record["verdict"])
                detail["resolved_by"] = f"{detail['eval_mode']}_llm_judge"
                resolved += 1
            elif str(record.get("raw_response", "")).startswith("LLM_JUDGE_ERROR"):
                detail["resolved_by"] = f"{detail['eval_mode']}_llm_judge_error"

        details.append(detail)

    total = len(details)
    correct = sum(1 for item in details if item["correct"])
    completed = sum(1 for item, row in zip(details, rows) if is_completed(row))
    metrics = {
        "total": total,
        "correct": correct,
        "error": total - correct,
        "accuracy": correct / total if total else 0.0,
        "completed_count": completed,
        "accuracy_completed": correct / completed if completed else 0.0,
        "static_correct": sum(1 for item in details if item["static_correct"]),
        "total_input_tokens": sum(item["input_tokens"] for item in details),
        "total_output_tokens": sum(item["output_tokens"] for item in details),
        "avg_input_token": (
            sum(item["input_tokens"] for item in details) / total if total else 0
        ),
        "avg_output_token": (
            sum(item["output_tokens"] for item in details) / total if total else 0
        ),
        "dataset_tag": inferred_tag,
        "llm_judge": {
            "enabled": enable_llm_judge,
            "available": judge is not None,
            "attempted": attempted,
            "resolved": resolved,
            "profile": judge_profile,
            "checkpoint_path": str(checkpoint_path),
        },
        "details": details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Metrics: {correct}/{total} correct, accuracy={metrics['accuracy']:.2%}")
    logger.info(f"Wrote metrics to {output_path}")
    return metrics
