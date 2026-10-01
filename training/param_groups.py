"""Splits a model's parameters into "decay" and "no-decay" sets for AdamW.

Standard practice (not something specific to this project) is to exclude norm-layer scale/offset
parameters and all biases from weight decay: shrinking a bias toward zero doesn't regularize
anything (it's a single per-channel offset, not a big matrix that can overfit), and the same is
true for BatchNorm/LayerNorm/GroupNorm's learned scale+offset. Decaying them anyway is not a
deliberate choice this project ever made — it was just what a single project-wide
`weight_decay=...` on every parameter does by default (see CLAUDE.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn

_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)


def no_decay_param_ids(model: nn.Module) -> set[int]:
    """`id()` of every parameter that should be exempt from weight decay: every bias in the
    model, plus the weight of every norm layer (its "bias" is already caught by the bias rule)."""
    ids: set[int] = set()
    for name, param in model.named_parameters():
        if name.endswith(".bias") or name == "bias":
            ids.add(id(param))
    for module in model.modules():
        if isinstance(module, _NORM_TYPES) and module.weight is not None:
            ids.add(id(module.weight))
    return ids


def split_decay(params, no_decay_ids: set[int]) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partitions an iterable of parameters into (decay, no_decay), preserving order."""
    decay, no_decay = [], []
    for p in params:
        (no_decay if id(p) in no_decay_ids else decay).append(p)
    return decay, no_decay


def build_param_groups(
    model: nn.Module, backbone_params, head_params, lr_backbone: float, lr_head: float, weight_decay: float
) -> list[dict]:
    """Builds up to 4 AdamW param groups (backbone/head x decay/no-decay), each named so callers
    can find "the backbone group" / "the head group" by name rather than by list index (a group
    can be empty and is then omitted, e.g. a backbone with no bias/norm params at all). No-decay
    groups get weight_decay=0.0 regardless of `weight_decay`; decay groups get exactly it."""
    no_decay_ids = no_decay_param_ids(model)
    backbone_decay, backbone_no_decay = split_decay(backbone_params, no_decay_ids)
    head_decay, head_no_decay = split_decay(head_params, no_decay_ids)

    groups = []
    for name, params, lr, wd in (
        ("backbone", backbone_decay, lr_backbone, weight_decay),
        ("backbone_no_decay", backbone_no_decay, lr_backbone, 0.0),
        ("head", head_decay, lr_head, weight_decay),
        ("head_no_decay", head_no_decay, lr_head, 0.0),
    ):
        if params:
            groups.append({"params": params, "lr": lr, "name": name, "weight_decay": wd})
    return groups
