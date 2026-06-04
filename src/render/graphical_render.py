from __future__ import annotations

import base64
import io
import json
import math
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI
from PIL import Image

from src.inference.evaluation import ModelConfig, load_model_config
from src.utils.token_sizing import find_nearest_valid_token, get_size_calculator


T2I_COMIC_PROMPT_TEMPLATE = """You are an expert educational illustrator.

Task
Create a compact, step-by-step comic-style illustration that explains how to solve a math problem.

Input
- Question: {problem}
- Solution (for reference only): {solution}

Strict requirements
1. The comic must follow the solution steps, but must not reveal the final answer explicitly.
2. Include only visuals that directly help explain the solving process.
   - Do not include irrelevant people, characters, decorations, or background scenery.
3. Break the solution into 2–4 clear logical steps, with one panel per step.
4. Each panel must:
   - Show the key transformation, setup, or reasoning step visually
   - Use very minimal text: short labels or hints only
   - Keep formulas and annotations concise
5. Do not show the final numerical or symbolic answer anywhere in the image.
6. Prioritize clarity, compactness, and readability over artistic detail.

Layout and compactness requirements
- Use a tight multi-panel layout: either
  - 2–4 panels in a single horizontal row, or
  - a compact 2×2 grid if needed.
- Keep panel spacing narrow and outer margins small.
- Make the content fill most of the canvas; avoid large empty areas.
- Use tight cropping and a dense educational layout.
- Keep each panel visually simple but information-rich.
- Maintain consistent panel size and alignment.
- Ensure the full comic looks compact, balanced, and easy to scan quickly.

Visual style
- Clean educational comic style
- Simple shapes, clear labels, high contrast
- Plain white background
- Minimal visual clutter
- No unrelated cartoon characters
- Crisp linework and easy-to-read math notation

Output format
- A single compact multi-panel comic illustration
- 2–4 panels total
- Each panel corresponds to one logical step in the solution
- Do not include the final answer

Make the composition extra compact: minimize whitespace, reduce gutters between panels, keep text extremely brief, and let the instructional content occupy most of the image area."""

VISUAL_TOKEN_PATCH_SIZE = 32
DEFAULT_T2I_PROFILE = "nano-banana-pro"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
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


def resize_with_padding(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_width, target_height), "white")
    canvas.paste(
        resized,
        ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2),
    )
    return canvas


@dataclass
class T2IImageResult:
    image_path: str
    status: str
    prompt: str = ""
    latency_ms: int = 0
    error_message: str = ""
    reasoning_token: int | None = None
    actual_tokens: int | None = None
    source_size: tuple[int, int] | None = None
    final_size: tuple[int, int] | None = None


class T2IImageGenerator:
    def __init__(
        self,
        profile_name: str = DEFAULT_T2I_PROFILE,
        size: str = "2:3",
        response_format: str = "b64_json",
        aspect_ratio: str | None = None,
        token_sizing_model: str = "qwen3-vl",
    ):
        config = load_model_config(profile_name)
        if config is None:
            raise ValueError(f"Missing image generation profile: {profile_name}")
        self.config: ModelConfig = config
        self.size = size
        self.response_format = response_format
        self.aspect_ratio = aspect_ratio
        self.token_sizing_model = token_sizing_model
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @staticmethod
    def build_prompt(entry: dict[str, Any]) -> str:
        problem = str(entry.get("problem", "") or "").strip()
        solution = str(entry.get("solution", "") or "").strip()
        return T2I_COMIC_PROMPT_TEMPLATE.format(problem=problem, solution=solution)

    @staticmethod
    def _entry_id(entry: dict[str, Any], index: int | None = None) -> str:
        raw_id = str(entry.get("id", "") or "").strip()
        if not raw_id and index is not None:
            raw_id = f"sample-{index}"
        if not raw_id:
            raw_id = "sample"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("._")
        return safe_id or "sample"

    @staticmethod
    def _first_image_item(response: Any) -> Any:
        return response.data[0]

    @staticmethod
    def _item_value(item: Any, key: str) -> str:
        return str(getattr(item, key, "") or "")

    @staticmethod
    def _image_suffix(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".png"

    @classmethod
    def _write_image_bytes(cls, image_bytes: bytes, output_stem: Path) -> Path:
        output_path = output_stem.with_suffix(cls._image_suffix(image_bytes))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        return output_path

    @staticmethod
    def _count_visual_tokens(
        image_size: tuple[int, int],
        patch_size: int = VISUAL_TOKEN_PATCH_SIZE,
    ) -> int:
        width, height = image_size
        return math.ceil(width / patch_size) * math.ceil(height / patch_size)

    @staticmethod
    def _normalize_reasoning_token_budget(reasoning_token: int, model_name: str) -> int:
        if reasoning_token <= 0:
            raise ValueError("reasoning_token must be positive")
        if "qwen" in model_name.lower():
            return find_nearest_valid_token(reasoning_token)
        return reasoning_token

    @staticmethod
    def _download_image_bytes(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()

    def _reasoning_token_for_entry(self, entry: dict[str, Any]) -> int | None:
        raw_tokens = entry.get("reasoning_token")
        if raw_tokens is None or raw_tokens == "":
            return None
        return int(raw_tokens)

    def _write_generated_image(
        self,
        image_bytes: bytes,
        output_stem: Path,
        original_output_stem: Path,
        reasoning_token: int | None,
    ) -> tuple[Path, dict[str, Any]]:
        self._write_image_bytes(image_bytes, original_output_stem)

        if reasoning_token is None:
            return self._write_image_bytes(image_bytes, output_stem), {}

        token_budget = self._normalize_reasoning_token_budget(
            reasoning_token,
            self.token_sizing_model,
        )
        with Image.open(io.BytesIO(image_bytes)) as image:
            source_size = image.size
            size_func = get_size_calculator(self.token_sizing_model)
            final_width, final_height = size_func(source_size, token_budget)
            resized = resize_with_padding(image.convert("RGB"), final_width, final_height)

        output_path = output_stem.with_suffix(self._image_suffix(image_bytes))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(output_path)
        final_size = (final_width, final_height)
        return output_path, {
            "reasoning_token": reasoning_token,
            "actual_tokens": self._count_visual_tokens(final_size),
            "source_size": source_size,
            "final_size": final_size,
        }

    def _generate_image(self, prompt: str) -> tuple[Any, int]:
        start = time.time()
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "size": self.size,
            "response_format": self.response_format,
        }
        if self.aspect_ratio:
            kwargs["extra_body"] = {"aspect_ratio": self.aspect_ratio}
        response = self.client.images.generate(**kwargs)
        latency_ms = int((time.time() - start) * 1000)
        return response, latency_ms

    def generate_entry(
        self,
        entry: dict[str, Any],
        output_base: str | Path,
        index: int | None = None,
    ) -> T2IImageResult:
        problem = str(entry.get("problem", "") or "").strip()
        if not problem:
            return T2IImageResult("", "missing_problem", error_message="entry has no problem")

        prompt = self.build_prompt(entry)
        output_root = Path(output_base)
        entry_id = self._entry_id(entry, index=index)
        image_stem = output_root / "images" / entry_id
        original_image_stem = output_root / "original_images" / entry_id
        try:
            reasoning_token = self._reasoning_token_for_entry(entry)
            response, latency_ms = self._generate_image(prompt)
            item = self._first_image_item(response)
            b64_json = self._item_value(item, "b64_json")
            if b64_json:
                image_path, resize_meta = self._write_generated_image(
                    base64.b64decode(b64_json),
                    image_stem,
                    original_image_stem,
                    reasoning_token,
                )
            else:
                url = self._item_value(item, "url")
                if not url:
                    raise ValueError("image generation response missing b64_json/url")
                image_path, resize_meta = self._write_generated_image(
                    self._download_image_bytes(url),
                    image_stem,
                    original_image_stem,
                    reasoning_token,
                )
        except Exception as exc:
            logger.warning("T2I image generation failed | id={} | error={}", entry.get("id"), exc)
            return T2IImageResult("", "failed", prompt=prompt, error_message=str(exc))

        return T2IImageResult(
            image_path=image_path.relative_to(output_root).as_posix(),
            status="generated",
            prompt=prompt,
            latency_ms=latency_ms,
            reasoning_token=resize_meta.get("reasoning_token"),
            actual_tokens=resize_meta.get("actual_tokens"),
            source_size=resize_meta.get("source_size"),
            final_size=resize_meta.get("final_size"),
        )


def generate_t2i_image_dataset(
    input_jsonl: str,
    output_jsonl: str,
    output_base: str | Path,
    profile_name: str = DEFAULT_T2I_PROFILE,
    sample_id: int | str | None = None,
    max_items: int | None = None,
    size: str = "2:3",
    response_format: str = "b64_json",
    aspect_ratio: str | None = None,
    token_sizing_model: str = "qwen3-vl",
) -> dict[str, int]:
    entries = read_jsonl(input_jsonl)
    if sample_id is not None:
        sample_id_text = str(sample_id)
        entries = [entry for entry in entries if str(entry.get("id", "")) == sample_id_text]
    if max_items is not None:
        entries = entries[:max_items]

    generator = T2IImageGenerator(
        profile_name=profile_name,
        size=size,
        response_format=response_format,
        aspect_ratio=aspect_ratio,
        token_sizing_model=token_sizing_model,
    )
    output_rows = []
    stats = {"entries": len(entries), "generated": 0, "failed": 0, "missing_problem": 0}
    for index, entry in enumerate(entries, start=1):
        result = generator.generate_entry(entry, output_base=output_base, index=index)
        row = dict(entry)
        row["image_path"] = result.image_path
        row["t2i_image_status"] = result.status
        row["t2i_image_model"] = generator.config.model
        row["t2i_image_latency_ms"] = result.latency_ms
        row["t2i_image_prompt"] = result.prompt
        if result.reasoning_token is not None:
            row["t2i_image_reasoning_token"] = result.reasoning_token
            row["t2i_image_actual_tokens"] = result.actual_tokens
            row["t2i_image_token_sizing_model"] = generator.token_sizing_model
            row["t2i_image_source_size"] = list(result.source_size or ())
            row["t2i_image_final_size"] = list(result.final_size or ())
        if result.error_message:
            row["t2i_image_error"] = result.error_message
        output_rows.append(row)
        if result.status == "generated":
            stats["generated"] += 1
        elif result.status == "missing_problem":
            stats["missing_problem"] += 1
        else:
            stats["failed"] += 1

    write_jsonl(output_jsonl, output_rows)
    return stats


__all__ = [
    "T2IImageGenerator",
    "T2IImageResult",
    "T2I_COMIC_PROMPT_TEMPLATE",
    "DEFAULT_T2I_PROFILE",
    "generate_t2i_image_dataset",
]
