from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from openai import OpenAI

from .base_predictor import BasePredictor


STRICT_ANSWER_SYSTEM_PROMPT = (
    "You are an expert problem solver. "
    "Your only task is to provide the final answer in \\boxed{ANSWER} format. "
    "Do not show your work, intermediate steps, or reasoning."
)

FREE_REASONING_SYSTEM_PROMPT = (
    "You are an expert problem solver. "
    "Solve the problem step by step. "
    "At the end, provide your final answer in the format \\boxed{ANSWER}."
)

SYSTEM_PROMPTS = {
    "no_reasoning": STRICT_ANSWER_SYSTEM_PROMPT,
    "text_reasoning": STRICT_ANSWER_SYSTEM_PROMPT,
    "img_reasoning": STRICT_ANSWER_SYSTEM_PROMPT,
    "free_reasoning": FREE_REASONING_SYSTEM_PROMPT,
}

DEFAULT_MAX_TOKENS = {
    "no_reasoning": 256,
    "free_reasoning": None,
    "text_reasoning": 256,
    "img_reasoning": 256,
}

REASONING_IMAGE_PREFIX = "reasoning_image_"


def infer_image_mime_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    raise ValueError(f"Unsupported image format: {suffix or '<missing suffix>'}")


def build_image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.is_dir():
        raise ValueError(f"Image path is a directory: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{infer_image_mime_type(path)};base64,{encoded}"


class Predictor(BasePredictor):
    """OpenAI-compatible predictor for the supported text and image modes."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        task_type: str = "no_reasoning",
        max_tokens: int | None = None,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        if task_type not in SYSTEM_PROMPTS:
            raise ValueError(f"Unsupported task_type: {task_type}")
        self.task_type = task_type
        self.max_tokens = DEFAULT_MAX_TOKENS[task_type] if max_tokens is None else max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def predict(self, input_data: dict[str, Any], prompt: str) -> dict[str, Any]:
        image_paths = self._collect_image_paths(input_data)
        if input_data.get("question_image") or image_paths:
            return self._predict_with_images(input_data, prompt)
        return self._predict_text(prompt)

    @staticmethod
    def _normalize_image_path_value(raw_value: Any) -> list[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, (list, tuple)):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        text = str(raw_value).strip()
        return [text] if text else []

    @staticmethod
    def _reasoning_image_sort_key(field_name: str) -> tuple[int, str]:
        suffix = field_name.removeprefix(REASONING_IMAGE_PREFIX)
        if suffix.isdigit():
            return int(suffix), ""
        return 10**9, field_name

    @classmethod
    def _collect_image_paths(cls, input_data: dict[str, Any]) -> list[str]:
        image_paths = cls._normalize_image_path_value(input_data.get("image_path"))
        if image_paths:
            return image_paths

        image_paths = cls._normalize_image_path_value(input_data.get("reasoning_images"))
        if image_paths:
            return image_paths

        field_names = sorted(
            (key for key in input_data if key.startswith(REASONING_IMAGE_PREFIX)),
            key=cls._reasoning_image_sort_key,
        )
        return [
            str(input_data[field_name]).strip()
            for field_name in field_names
            if str(input_data.get(field_name) or "").strip()
        ]

    def _completion_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0 if self.temperature is None else self.temperature,
        }
        if self.max_tokens is not None:
            key = "max_completion_tokens" if "gpt-5.1" in self.model.lower() else "max_tokens"
            args[key] = self.max_tokens
        if self.reasoning_effort:
            args["reasoning_effort"] = self.reasoning_effort
        return args

    def _predict_text(self, prompt: str) -> dict[str, Any]:
        completion = self.client.chat.completions.create(
            **self._completion_args(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[self.task_type]},
                {"role": "user", "content": prompt},
            ],
        )
        return self._format_completion(completion)

    def _predict_with_images(self, input_data: dict[str, Any], prompt: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        question_image = input_data.get("question_image")
        if question_image:
            content.extend(
                [
                    {"type": "text", "text": "Question image:"},
                    {"type": "image_url", "image_url": {"url": build_image_data_url(question_image)}},
                ]
            )

        for index, image_path in enumerate(self._collect_image_paths(input_data), start=1):
            label = "Reasoning image:" if index == 1 else f"Reasoning image {index}:"
            content.extend(
                [
                    {"type": "text", "text": label},
                    {"type": "image_url", "image_url": {"url": build_image_data_url(image_path)}},
                ]
            )

        if len(content) == 1:
            raise ValueError("No image was provided for multimodal inference")

        completion = self.client.chat.completions.create(
            **self._completion_args(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[self.task_type]},
                {"role": "user", "content": content},
            ],
        )
        return self._format_completion(completion)

    @staticmethod
    def _format_completion(completion: Any) -> dict[str, Any]:
        usage = getattr(completion, "usage", None)
        return {
            "prediction": completion.choices[0].message.content or "",
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
