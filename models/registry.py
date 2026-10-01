"""Model registry: `@register_model("name")` + `build_model(cfg)`. Adding a new backbone is one
registry entry (a new factory function + a new experiment YAML `model:` block referencing it),
not a change anywhere else in the codebase.

Contract every registered model must satisfy: `forward(pixel_values)` returns raw multi-channel
logits `(B, num_classes, H, W)` at the SAME spatial resolution as the input — this uniform
contract is what makes losses/metrics/sliding-window inference backbone-agnostic with zero
special-casing (see inference/sliding_window.py).
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from configs.schema import ModelConfig

_REGISTRY: dict[str, Callable[[ModelConfig], nn.Module]] = {}


def register_model(name: str):
    def deco(factory: Callable[[ModelConfig], nn.Module]):
        if name in _REGISTRY:
            raise ValueError(f"Model {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name not in _REGISTRY:
        raise KeyError(f"Unknown model {cfg.name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](cfg)


def registered_models() -> list[str]:
    return sorted(_REGISTRY)
