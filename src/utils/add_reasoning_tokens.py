"""Add text and referenced-image token counts to reasoning data."""

from __future__ import annotations

import argparse
import json
import re
from math import ceil
from pathlib import Path
from typing import Any, Protocol

import tiktoken
from PIL import Image


PATCH_SIZE = 32
REASONING_IMAGE_PREFIX = "reasoning_image_"


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


def _references_image(solution: str, field_name: str, raw_path: Any) -> bool:
    if not raw_path:
        return False

    path = str(raw_path).strip().replace("\\", "/")
    references = {field_name, path, path.removeprefix("./"), f"./{path}"}
    return any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(reference)}(?![A-Za-z0-9_])",
            solution,
        )
        for reference in references
        if reference
    )


def _resolve_image_path(raw_path: Any, dataset_dir: Path) -> Path:
    image_path = Path(str(raw_path))
    if image_path.is_absolute():
        return image_path

    for base_dir in (dataset_dir, *dataset_dir.parents):
        candidate = (base_dir / image_path).resolve()
        if candidate.exists():
            return candidate
    return (dataset_dir / image_path).resolve()


def count_image_tokens(image_path: str | Path, patch_size: int = PATCH_SIZE) -> int:
    """Count ceil-based visual patches for one image."""
    with Image.open(image_path) as image:
        return ceil(image.width / patch_size) * ceil(image.height / patch_size)


def count_reasoning_tokens(
    entry: dict[str, Any],
    encoder: Encoder,
    dataset_dir: str | Path,
) -> int:
    """Count solution text tokens plus referenced reasoning-image tokens."""
    solution = str(entry.get("solution") or "")
    total = len(encoder.encode(solution))
    dataset_dir = Path(dataset_dir)

    for field_name, raw_path in entry.items():
        if not field_name.startswith(REASONING_IMAGE_PREFIX):
            continue
        if not _references_image(solution, field_name, raw_path):
            continue

        image_path = _resolve_image_path(raw_path, dataset_dir)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Reasoning image field '{field_name}' points to a missing file: {image_path}"
            )
        total += count_image_tokens(image_path)

    return total


def add_reasoning_tokens(
    input_path: str | Path,
    output_path: str | Path | None = None,
    encoding_name: str = "cl100k_base",
    encoder: Encoder | None = None,
) -> Path:
    """Write JSONL records with an updated ``reasoning_token`` field."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    encoder = encoder or tiktoken.get_encoding(encoding_name)

    entries = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for entry in entries:
        entry["reasoning_token"] = count_reasoning_tokens(
            entry,
            encoder,
            input_path.resolve().parent,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--encoding", default="cl100k_base")
    args = parser.parse_args()
    add_reasoning_tokens(args.input_path, args.output_path, args.encoding)


if __name__ == "__main__":
    main()
