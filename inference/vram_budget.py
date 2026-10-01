"""VRAM/RAM budget guard for sliding-window inference over large orthophotos.

Carried over in spirit from a sibling project's `orthophoto_inference.py`: refuse to allocate a
full-resolution probability accumulator past a memory budget unless explicitly overridden or an
automatic downsample is applied.
"""
from __future__ import annotations


def estimate_accumulator_gb(num_classes: int, width: int, height: int, dtype_bytes: int = 4) -> float:
    """Two float32 (C, H, W) accumulators (prob_sum, count broadcastable) plus a bit of headroom."""
    prob_sum_bytes = num_classes * height * width * dtype_bytes
    count_bytes = height * width * dtype_bytes  # count is (1, H, W), broadcast across classes
    return (prob_sum_bytes + count_bytes) * 1.1 / 1e9  # 10% headroom


def check_budget(num_classes: int, width: int, height: int, max_memory_gb: float, allow_large_memory: bool = False) -> None:
    estimated = estimate_accumulator_gb(num_classes, width, height)
    if estimated > max_memory_gb and not allow_large_memory:
        raise MemoryError(
            f"Estimated accumulator size {estimated:.2f}GB exceeds max_memory_gb={max_memory_gb}. "
            f"Pass --allow_large_memory to override, or reduce --max_dimension to downsample the "
            f"orthophoto before tiling, or reduce tile overlap."
        )
