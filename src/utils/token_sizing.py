"""Image-size calculators for model-specific visual token budgets."""

import math
from typing import Callable


SizeCalculator = Callable[[tuple[int, int], int], tuple[int, int]]


def _factorize_tokens(
    orig_w: int,
    orig_h: int,
    target_tokens: int,
    patch_size: int,
    min_patches: int = 4,
    max_ratio: float = 4.0,
) -> tuple[int, int]:
    """Choose the closest valid grid for a token budget."""
    target_ratio = orig_h / orig_w
    best_w_grid = best_h_grid = None
    min_log_diff = float("inf")

    limit = int(math.isqrt(target_tokens))
    for w in range(1, limit + 1):
        if target_tokens % w != 0:
            continue
        h = target_tokens // w
        for w_grid, h_grid in ((w, h), (h, w)):
            if w_grid < min_patches or h_grid < min_patches:
                continue
            if max(w_grid / h_grid, h_grid / w_grid) > max_ratio:
                continue
            ratio = h_grid / w_grid
            diff = abs(math.log(ratio) - math.log(target_ratio))
            if diff < min_log_diff:
                min_log_diff = diff
                best_w_grid = w_grid
                best_h_grid = h_grid

    if best_w_grid is None:
        best_w_grid = best_h_grid = min_patches

    return best_w_grid * patch_size, best_h_grid * patch_size


def has_valid_token_factorization(
    target_tokens: int,
    min_patches: int = 4,
    max_aspect_ratio: float = 4.0,
) -> bool:
    """Check whether a Qwen-style token budget has a valid patch grid."""
    if target_tokens < min_patches * min_patches:
        return False

    limit = int(math.isqrt(target_tokens))
    for w in range(min_patches, limit + 1):
        if target_tokens % w != 0:
            continue
        h = target_tokens // w
        if h < min_patches:
            continue
        if max(w / h, h / w) <= max_aspect_ratio:
            return True
    return False


def find_nearest_valid_token(
    target_token: int,
    min_patches: int = 4,
    max_aspect_ratio: float = 4.0,
    max_offset: int = 100,
) -> int:
    """Snap a token budget to the nearest valid Qwen-style factorization."""
    min_tokens = min_patches * min_patches
    if target_token < min_tokens:
        return min_tokens

    if has_valid_token_factorization(target_token, min_patches, max_aspect_ratio):
        return target_token

    for offset in range(1, max_offset + 1):
        larger = target_token + offset
        if has_valid_token_factorization(larger, min_patches, max_aspect_ratio):
            return larger

        smaller = target_token - offset
        if smaller >= min_tokens and has_valid_token_factorization(
            smaller,
            min_patches,
            max_aspect_ratio,
        ):
            return smaller

    return min_tokens


def qwen_size_for_tokens(
    orig_size: tuple[int, int],
    target_tokens: int,
    patch_size: int = 32,
) -> tuple[int, int]:
    """Qwen uses exact patch-grid factorization."""
    return _factorize_tokens(orig_size[0], orig_size[1], target_tokens, patch_size)


def gpt51_size_for_tokens(
    orig_size: tuple[int, int],
    target_tokens: int,
) -> tuple[int, int]:
    """Map GPT-5.1 visual token budgets to one of the supported image sizes."""
    token_to_sizes = {
        210: [(512, 512)],
        350: [(512, 1024), (1024, 512)],
        490: [(512, 1536), (1536, 512)],
        630: [(512, 2048), (768, 1024), (2048, 512)],
        910: [(768, 1536), (1536, 768)],
        1190: [(768, 2048), (2048, 768)],
    }

    available_tokens = sorted(token_to_sizes)
    chosen_token = available_tokens[-1]
    for available in available_tokens:
        if target_tokens <= available:
            chosen_token = available
            break

    layouts = token_to_sizes[chosen_token]
    orig_w, orig_h = orig_size
    orig_ratio = orig_w / orig_h if orig_h else 1.0
    want_landscape = orig_w >= orig_h

    def is_landscape(size: tuple[int, int]) -> bool:
        return size[0] >= size[1]

    candidates = [size for size in layouts if is_landscape(size) == want_landscape]
    if not candidates:
        candidates = layouts

    def score(size: tuple[int, int]) -> tuple[float, int]:
        w, h = size
        ratio_diff = abs((w / h if h else 1.0) - orig_ratio)
        return ratio_diff, -min(w, h)

    return min(candidates, key=score)


def get_size_calculator(model_name: str) -> SizeCalculator:
    """Return the size calculator for a model family."""
    model_name = model_name.lower()
    if "gpt" in model_name:
        return gpt51_size_for_tokens
    return qwen_size_for_tokens


def get_size_func(model_name: str) -> SizeCalculator:
    """Backward-compatible alias for older imports."""
    return get_size_calculator(model_name)


__all__ = [
    "find_nearest_valid_token",
    "get_size_calculator",
    "get_size_func",
    "gpt51_size_for_tokens",
    "has_valid_token_factorization",
    "qwen_size_for_tokens",
]
