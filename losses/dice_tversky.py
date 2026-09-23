"""Multi-label soft Dice / Tversky loss, per class channel."""
from __future__ import annotations

import torch


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor | float = 0.5,
    beta: torch.Tensor | float = 0.5,
    validity_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """logits/target: (B, C, H, W). alpha weights false positives, beta weights false negatives
    (alpha=beta=0.5 reduces to standard Dice). alpha/beta: scalar or (C,) tensor. Returns a (C,)
    tensor of per-class loss (1 - Tversky index).

    Masking both `prob` and `target` by `validity_mask` before computing TP/FP/FN is sufficient
    (no separate denominator bookkeeping needed, unlike focal loss): an invalid pixel with
    prob=0, target=0 contributes 0 to TP, FP, and FN alike, regardless of what the raw prediction
    would have been there.
    """
    prob = torch.sigmoid(logits)
    if validity_mask is not None:
        prob = prob * validity_mask
        target = target * validity_mask

    dims = (0, 2, 3)
    tp = (prob * target).sum(dims)  # (C,)
    fp = (prob * (1 - target)).sum(dims)
    fn = ((1 - prob) * target).sum(dims)

    tversky_index = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky_index


def dice_loss(logits: torch.Tensor, target: torch.Tensor, validity_mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Standard Dice loss = Tversky with alpha=beta=0.5."""
    return tversky_loss(logits, target, alpha=0.5, beta=0.5, validity_mask=validity_mask, eps=eps)
