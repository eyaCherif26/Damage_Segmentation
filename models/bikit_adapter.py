"""Loads a bikit-toolkit DaclNet classifier checkpoint's encoder into a segmentation model.

Confirmed checkpoint structure (read `port_damage_EC/bikit_models/models/daclnet.py` directly):
    {"base": str, "resolution": int, "hidden_layers": list[int], "num_class": int,
     "drop_prob": float, "state_dict": OrderedDict, "cat_to_name": dict}

`state_dict` keys are namespaced `base.*` (the encoder) and `classifier.*` (the whole-image
classification head — discarded here; this is the "reuse the encoder under a segmentation
decoder" case PROMPT.md asks for). `base` names one of {"mobilenet", "resnet", "efficientnet",
"mobilenetv2"} (torchvision/efficientnet_pytorch backbones) — NOT the same implementation as
smp's `tu-*` (timm) encoders used elsewhere in this project, so key names only partially overlap.
Enforces a >=50% tensor-coverage guard before accepting the load, adapted from a pattern in a
sibling project (`SegF_model/classification_backbones.py::_load_state_dict_flexible`) — aborts
loudly rather than silently training from a near-random backbone.
"""
from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn as nn


class CoverageError(RuntimeError):
    pass


def load_bikit_encoder_into(
    model: nn.Module,
    checkpoint_path: str,
    min_coverage: float = 0.5,
    key_rewrites: Sequence[tuple[str, str]] = (),
) -> dict:
    """Loads the `base.*` (encoder) tensors from a DaclNet checkpoint into `model`'s state dict
    wherever key names/shapes match, ignoring `classifier.*`. Raises CoverageError if fewer than
    `min_coverage` of the checkpoint's encoder tensors found a shape-matching home in `model`.

    `key_rewrites` are (regex, replacement) pairs producing extra candidate spellings for the
    checkpoint's keys, for backbones whose wrapper dropped a module name — e.g. bikit stores
    torchvision MobileNet as `nn.Sequential(features, avgpool)`, so `0.4.block...` has to be tried
    as `features.4.block...` too.

    Returns a small report dict for logging (checkpoint metadata + coverage stats).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_state = checkpoint["state_dict"]
    encoder_tensors = {k[len("base.") :]: v for k, v in ckpt_state.items() if k.startswith("base.")}
    if not encoder_tensors:
        raise CoverageError(f"No 'base.*' (encoder) tensors found in checkpoint {checkpoint_path}")

    model_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for ckpt_key, ckpt_tensor in encoder_tensors.items():
        candidates = [ckpt_key] + [re.sub(pattern, repl, ckpt_key) for pattern, repl in key_rewrites]
        for model_key, model_tensor in model_state.items():
            if model_tensor.shape == ckpt_tensor.shape and any(model_key.endswith(c) for c in candidates):
                matched[model_key] = ckpt_tensor
                break

    coverage = len(matched) / len(encoder_tensors)
    if coverage < min_coverage:
        raise CoverageError(
            f"bikit checkpoint {checkpoint_path!r} (base={checkpoint.get('base')!r}) matched only "
            f"{coverage:.1%} of its encoder tensors against the target model's state dict "
            f"(need >= {min_coverage:.0%}). Aborting rather than silently training from a "
            "near-random backbone — the bikit base architecture and this project's encoder "
            "implementation likely don't share key names/shapes closely enough for a direct load."
        )

    model_state.update(matched)
    model.load_state_dict(model_state)

    return {
        "checkpoint_path": checkpoint_path,
        "base": checkpoint.get("base"),
        "resolution": checkpoint.get("resolution"),
        "coverage": coverage,
        "matched_tensors": len(matched),
        "total_encoder_tensors": len(encoder_tensors),
    }
