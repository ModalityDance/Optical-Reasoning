"""Self-contained XeLaTeX dense renderer for typographic reasoning images."""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from subprocess import run
from typing import Optional

import fitz
from PIL import Image, ImageOps

from src.utils.token_sizing import find_nearest_valid_token, get_size_calculator


@dataclass(frozen=True)
class DenseLatexLayout:
    font_size: int
    text_width_in: float
    edge_blank_ratio: float
    fill_ratio: float
    score: float
    source_size: tuple[int, int]
    final_size: tuple[int, int]


@dataclass(frozen=True)
class DenseLatexRenderResult:
    output_path: str
    font_size: int
    text_width_in: float
    edge_blank_ratio: float
    fill_ratio: float
    score: float
    source_size: tuple[int, int]
    final_size: tuple[int, int]
    search_strategy: str = "full_search"
    render_attempt_count: int = 0


def _preprocess_latex_text(text: str) -> str:
    text = re.sub(r"\$\$([^\$]+?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)
    text = re.sub(r"\\Box\b", r"\\mysquare", text)
    text = re.sub(r"\\square\b", r"\\mysquare", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    return text


def _strip_latex_for_stats(text: str) -> str:
    text = re.sub(r"\\\[[\s\S]*?\\\]", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text).strip()


def _choose_text_width_in(text: str, min_width_in: float, max_width_in: float) -> float:
    longest = max(
        (len(cleaned) for line in text.splitlines() if (cleaned := _strip_latex_for_stats(line))),
        default=0,
    )
    if longest <= 60:
        width = max_width_in
    elif longest <= 90:
        width = max_width_in - 0.5
    elif longest <= 120:
        width = max_width_in - 1.0
    else:
        width = min_width_in
    return max(min_width_in, min(max_width_in, width))


def _width_candidates(text: str, min_text_width_in: float, max_text_width_in: float) -> list[float]:
    base = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0]
    low = min(min_text_width_in, max_text_width_in)
    high = max(min_text_width_in, max_text_width_in)
    selected = [width for width in base if low <= width <= high] or [round(low, 2)]
    selected.extend([round(low, 2), round(high, 2)])
    preferred = _choose_text_width_in(text, low, high)
    return sorted(set(selected), key=lambda width: (abs(width - preferred), -width))


def _build_latex_document(text: str, font_size: int, text_width_in: float) -> str:
    baselineskip = font_size * 1.25
    return "\n".join(
        [
            r"\documentclass[12pt]{article}",
            r"\usepackage{fontspec}",
            r"\usepackage{unicode-math}",
            r"\setmainfont{Latin Modern Roman}",
            r"\setmathfont{Latin Modern Math}",
            r"\usepackage{amsmath,amsthm,mathtools}",
            r"\usepackage{xcolor}",
            r"\usepackage[papersize={7in,20in},textwidth={"
            + f"{text_width_in}in"
            + r"},margin=0.1in]{geometry}",
            r"\providecommand{\mysquare}{\mathbin{\vcenter{\hbox{\rule{3pt}{3pt}}}}}",
            r"\pagestyle{empty}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.5em}",
            r"\sloppy",
            r"\tolerance=1000",
            r"\emergencystretch=1em",
            r"\begin{document}",
            f"\\fontsize{{{font_size}}}{{{baselineskip}}}\\selectfont",
            _preprocess_latex_text(text),
            r"\end{document}",
        ]
    )


def _compile_latex(latex_content: str, output_dir: Path, timeout: int) -> Path:
    tex_path = output_dir / "render.tex"
    tex_path.write_text(latex_content, encoding="utf-8")
    result = run(
        [
            "xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output_dir}",
            str(tex_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=output_dir,
    )
    pdf_path = output_dir / "render.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"xelatex failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    return pdf_path


def _pdf_to_image(pdf_path: Path, dpi: int) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    if len(doc) == 0:
        doc.close()
        raise RuntimeError("xelatex produced an empty PDF")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    page_images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        page_images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()

    width = max(image.width for image in page_images)
    height = sum(image.height for image in page_images)
    merged = Image.new("RGB", (width, height), "white")
    y = 0
    for image in page_images:
        merged.paste(image, ((width - image.width) // 2, y))
        y += image.height
    return merged


def _crop_whitespace(image: Image.Image, pad: int) -> Image.Image:
    bbox = ImageOps.invert(image.convert("L")).getbbox()
    if bbox is None:
        return image
    left, upper, right, lower = bbox
    return image.crop(
        (
            max(0, left - pad),
            max(0, upper - pad),
            min(image.width, right + pad),
            min(image.height, lower + pad),
        )
    )


def _resize_with_padding(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    orig_w, orig_h = image.size
    ratio = orig_w / orig_h
    target_ratio = target_w / target_h
    if ratio > target_ratio:
        new_w = target_w
        new_h = max(1, int(target_w / ratio))
    else:
        new_h = target_h
        new_w = max(1, int(target_h * ratio))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), "white")
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def _render_latex_to_image(
    text: str,
    font_size: int,
    text_width_in: float,
    dpi: int,
    crop_pad: int,
    timeout: int,
) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_path = _compile_latex(
            _build_latex_document(text, font_size, text_width_in),
            tmp_path,
            timeout,
        )
        image = _pdf_to_image(pdf_path, dpi)
    return _crop_whitespace(image, crop_pad)


def _image_density_metrics(image: Image.Image) -> tuple[float, float]:
    bbox = ImageOps.invert(image.convert("L")).getbbox()
    if bbox is None:
        return 1.0, 0.0
    left, upper, right, lower = bbox
    edge_blank_ratio = max(left, upper, image.width - right, image.height - lower) / max(
        image.width,
        image.height,
    )
    fill_ratio = ((right - left) * (lower - upper)) / max(1, image.width * image.height)
    return edge_blank_ratio, fill_ratio


def _layout_score(
    image: Image.Image,
    edge_blank_ratio: float,
    fill_ratio: float,
    target_w: int,
    target_h: int,
    max_edge_blank_ratio: float,
) -> float:
    bbox = ImageOps.invert(image.convert("L")).getbbox()
    if bbox is None:
        return -1.0
    left, upper, right, lower = bbox
    wr = max(1, right - left) / max(1, target_w)
    hr = max(1, lower - upper) / max(1, target_h)
    shape_penalty = abs(math.log(max(wr, 1e-6) / max(hr, 1e-6)))
    return fill_ratio - 0.08 * shape_penalty - 0.20 * max(0.0, edge_blank_ratio - max_edge_blank_ratio)


def _layout_from_image(
    image: Image.Image,
    font_size: int,
    text_width_in: float,
    final_width: int,
    final_height: int,
    max_edge_blank_ratio: float,
) -> tuple[DenseLatexLayout, Image.Image]:
    final_image = _resize_with_padding(image, final_width, final_height)
    edge_blank_ratio, fill_ratio = _image_density_metrics(final_image)
    score = _layout_score(final_image, edge_blank_ratio, fill_ratio, final_width, final_height, max_edge_blank_ratio)
    return (
        DenseLatexLayout(
            font_size=font_size,
            text_width_in=text_width_in,
            edge_blank_ratio=edge_blank_ratio,
            fill_ratio=fill_ratio,
            score=score,
            source_size=image.size,
            final_size=(final_width, final_height),
        ),
        final_image,
    )


def _font_search_plan(min_font_size: int, max_font_size: int) -> tuple[list[int], list[int]]:
    coarse = list(range(max_font_size, min_font_size - 1, -2))
    if coarse and coarse[-1] != min_font_size:
        coarse.append(min_font_size)
    coarse_set = set(coarse)
    fine = [size for size in range(max_font_size, min_font_size - 1, -1) if size not in coarse_set]
    return coarse, fine


def _normalize_reasoning_token(reasoning_token: int, model_name: str) -> int:
    if "qwen" in model_name.lower():
        return find_nearest_valid_token(reasoning_token)
    return reasoning_token


def render_dense_latex(
    text: str,
    output_path: str,
    reasoning_token: int,
    model_name: str = "qwen3-vl",
    dpi: int = 150,
    crop_pad: int = 10,
    min_font_size: int = 8,
    max_font_size: int = 20,
    min_text_width_in: float = 1.8,
    max_text_width_in: float = 6.0,
    timeout: int = 30,
    target_fill_ratio: float = 0.70,
    max_edge_blank_ratio: float = 0.10,
    **_: object,
) -> DenseLatexRenderResult:
    """Render LaTeX with XeLaTeX, fit it to the reasoning-token image budget, and save PNG."""
    text = text if text and text.strip() else " "
    target_tokens = _normalize_reasoning_token(int(reasoning_token), model_name)
    width_candidates = _width_candidates(text, min_text_width_in, max_text_width_in)
    base_font_size = min(max_font_size, max(min_font_size, 12))

    base_image: Optional[Image.Image] = None
    base_width = width_candidates[0]
    last_error: Exception | None = None
    for text_width_in in width_candidates:
        try:
            base_image = _render_latex_to_image(text, base_font_size, text_width_in, dpi, crop_pad, timeout)
            base_width = text_width_in
            break
        except Exception as exc:
            last_error = exc

    if base_image is None:
        raise RuntimeError(f"Failed to render base LaTeX image: {last_error}")

    final_width, final_height = get_size_calculator(model_name)(base_image.size, target_tokens)
    best_layout, best_image = _layout_from_image(
        base_image,
        base_font_size,
        base_width,
        final_width,
        final_height,
        max_edge_blank_ratio,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_image.save(path, "PNG")
    return DenseLatexRenderResult(
        output_path=str(path),
        font_size=best_layout.font_size,
        text_width_in=best_layout.text_width_in,
        edge_blank_ratio=best_layout.edge_blank_ratio,
        fill_ratio=best_layout.fill_ratio,
        score=best_layout.score,
        source_size=best_layout.source_size,
        final_size=best_layout.final_size,
        search_strategy="probe_only",
        render_attempt_count=1,
    )


def render_dense_latex_to_png(
    text: str,
    output_path: str,
    reasoning_token: int,
    model_name: str = "qwen3-vl",
    **kwargs: object,
) -> tuple[int, int]:
    return render_dense_latex(
        text=text,
        output_path=output_path,
        reasoning_token=reasoning_token,
        model_name=model_name,
        **kwargs,
    ).final_size


__all__ = [
    "DenseLatexLayout",
    "DenseLatexRenderResult",
    "render_dense_latex",
    "render_dense_latex_to_png",
]
