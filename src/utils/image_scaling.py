"""Resize scale=1 images to a target area scale."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Return dimensions whose pixel area is multiplied by ``scale``."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if scale <= 0:
        raise ValueError("Scale must be positive")

    edge_scale = math.sqrt(scale)
    return (
        max(1, round(width * edge_scale)),
        max(1, round(height * edge_scale)),
    )


def resize_image(
    source_path: str | Path,
    output_path: str | Path,
    scale: float,
) -> Path:
    """Resize one scale=1 image and save it at the target area scale."""
    source_path = Path(source_path)
    output_path = Path(output_path)

    with Image.open(source_path) as image:
        target_size = scaled_size(image.width, image.height, scale)
        resized = image.resize(target_size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(output_path)

    return output_path


def resize_directory(
    source_dir: str | Path,
    output_dir: str | Path,
    scale: float,
    pattern: str = "*.png",
    workers: int = 8,
) -> list[Path]:
    """Resize matching scale=1 images while preserving relative paths."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_paths = sorted(source_dir.rglob(pattern))

    def resize(source_path: Path) -> Path:
        destination = output_dir / source_path.relative_to(source_dir)
        return resize_image(source_path, destination, scale)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(resize, source_paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize scale=1 images to a target pixel-area scale."
    )
    parser.add_argument("source_dir", type=Path, help="Directory containing scale=1 images.")
    parser.add_argument("output_dir", type=Path, help="Directory for resized images.")
    parser.add_argument("--scale", type=float, required=True, help="Target area scale.")
    parser.add_argument("--pattern", default="*.png", help="Recursive input glob pattern.")
    parser.add_argument("--workers", type=int, default=8, help="Number of resize workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("Workers must be positive")

    outputs = resize_directory(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        pattern=args.pattern,
        workers=args.workers,
    )
    print(f"Resized {len(outputs)} images to area scale={args.scale}")


if __name__ == "__main__":
    main()
