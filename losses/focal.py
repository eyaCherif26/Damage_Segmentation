"""Per-class binary focal loss, one term per mask channel (no softmax anywhere — multi-label)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor | float = 0.25,
    gamma: torch.Tensor | float = 2.0,
    validity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """logits/target: (B, C, H, W). alpha/gamma: scalar or (C,) tensor (broadcast per class).
    validity_mask: (B, 1, H, W) or None (all-valid). Returns a (C,) tensor of per-class mean loss
    over all valid pixels (never averaged across an empty/all-invalid class silently as 0 — see
    the clamp on the denominator, which returns 0 only when truly no valid pixels exist)."""
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.view(1, -1, 1, 1)
    if isinstance(gamma, torch.Tensor):
        gamma = gamma.view(1, -1, 1, 1)

    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t) ** gamma * ce  # (B, C, H, W)

    if validity_mask is not None:
        loss = loss * validity_mask
        denom = validity_mask.expand_as(loss).sum(dim=(0, 2, 3)).clamp_min(1.0)
    else:
        denom = torch.full((loss.shape[1],), loss.shape[0] * loss.shape[2] * loss.shape[3], device=loss.device, dtype=loss.dtype)

    return loss.sum(dim=(0, 2, 3)) / denom
