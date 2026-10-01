"""Soft clDice loss (Shit, Paetzold, Sekuboyina, Ezhov, Unger, Zhylka, Pluim, Bauer, Menze, "clDice — A Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation," CVPR 2021) `[Literature]`.

Applied only to `thin_structure_classes` (Cracks) — see CLAUDE.md/docs/methodology.md for why
this is the chosen default over boundary-F1 for the loss term: it specifically rewards
centerline/topological connectivity under small positional offsets, the exact failure mode of
plain pixel-IoU-style losses on thin, elongated structures.

Differentiable soft-skeletonization via iterative morphological soft-erode/soft-dilate
(max-pooling based), following the reference formulation from the paper.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """img: (B, 1, H, W) in [0, 1]. Returns a same-shape soft skeleton."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(num_iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cl_dice_loss(
    logits_channel: torch.Tensor,
    target_channel: torch.Tensor,
    validity_mask: torch.Tensor | None = None,
    num_iters: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single-channel (one class) soft clDice loss. logits_channel/target_channel: (B, 1, H, W).
    Returns a scalar."""
    prob = torch.sigmoid(logits_channel)
    if validity_mask is not None:
        prob = prob * validity_mask
        target_channel = target_channel * validity_mask

    pred_skel = soft_skeletonize(prob, num_iters)
    target_skel = soft_skeletonize(target_channel, num_iters)

    t_prec = (torch.sum(pred_skel * target_channel) + eps) / (torch.sum(pred_skel) + eps)
    t_sens = (torch.sum(target_skel * prob) + eps) / (torch.sum(target_skel) + eps)
    cl_dice = 2.0 * t_prec * t_sens / (t_prec + t_sens + eps)
    return 1.0 - cl_dice
