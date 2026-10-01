"""Resolves AMP settings (autocast dtype, whether autocast/GradScaler run) from config + device,
factored out of Trainer.__init__ so it's unit-testable without a real CUDA device (this project's
tests run CPU-only, CUDA_VISIBLE_DEVICES="").

bf16 is the global default (`configs/schema.py::TrainingConfig.amp_dtype`) because it has fp32's
exponent range and cannot overflow the way fp16 can at ~65504 — the root cause of a real training
incident (an fp16-autocast activation overflow permanently poisoned a BatchNorm running-stat
buffer; see CLAUDE.md's Pitfalls). It can be disabled per-experiment (`amp_dtype: fp16` in that
YAML) to reproduce the old behavior or on hardware without native bf16 support.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AmpSettings:
    enabled: bool  # whether autocast actually runs (always False off-CUDA, regardless of `amp`)
    autocast_dtype: torch.dtype
    scaler_enabled: bool  # GradScaler is only meaningful for fp16; always False for bf16
    warning: str | None = None  # set when bf16 was requested on hardware without native support


def resolve_amp(amp: bool, amp_dtype: str, device_type: str, bf16_supported: bool) -> AmpSettings:
    if amp_dtype not in ("bf16", "fp16"):
        raise ValueError(f"amp_dtype must be 'bf16' or 'fp16', got {amp_dtype!r}")
    enabled = amp and device_type == "cuda"
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler_enabled = enabled and amp_dtype == "fp16"
    warning = None
    if enabled and amp_dtype == "bf16" and not bf16_supported:
        warning = (
            "training.amp_dtype='bf16' but this GPU has no native bf16 support (pre-Ampere) — it "
            "will still run (software fallback) but may be slower than fp16. Set "
            "training.amp_dtype: fp16 to use the accelerated path on this hardware."
        )
    return AmpSettings(enabled, autocast_dtype, scaler_enabled, warning)
