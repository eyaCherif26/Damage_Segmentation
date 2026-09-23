"""Config-driven weighted combination: per-class focal + Dice/Tversky (+ optional clDice on
thin-structure classes), validity-mask aware throughout."""
from __future__ import annotations

import torch
import torch.nn as nn

from configs.schema import LossConfig
from losses.boundary import soft_cl_dice_loss
from losses.dice_tversky import dice_loss, tversky_loss
from losses.focal import binary_focal_loss


class CombinedLoss(nn.Module):
    def __init__(self, cfg: LossConfig, mvp_classes: list[str], thin_structure_classes: list[str]):
        super().__init__()
        self.cfg = cfg
        self.mvp_classes = mvp_classes
        self.thin_structure_indices = [mvp_classes.index(c) for c in thin_structure_classes if c in mvp_classes]

        alpha = torch.tensor([cfg.per_class_alpha.get(c, cfg.default_alpha) for c in mvp_classes])
        gamma = torch.tensor([cfg.per_class_gamma.get(c, cfg.default_gamma) for c in mvp_classes])
        self.register_buffer("focal_alpha", alpha)
        self.register_buffer("focal_gamma", gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, validity_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        focal_per_class = binary_focal_loss(logits, target, self.focal_alpha, self.focal_gamma, validity_mask)

        if self.cfg.dice_variant == "tversky":
            dice_per_class = tversky_loss(logits, target, self.cfg.tversky_alpha, self.cfg.tversky_beta, validity_mask)
        elif self.cfg.dice_variant == "dice":
            dice_per_class = dice_loss(logits, target, validity_mask)
        else:
            raise ValueError(f"Unknown loss.dice_variant: {self.cfg.dice_variant!r}")

        total = self.cfg.focal_weight * focal_per_class.mean() + self.cfg.dice_weight * dice_per_class.mean()

        boundary_term = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.cfg.boundary_weight > 0 and self.thin_structure_indices:
            terms = []
            for idx in self.thin_structure_indices:
                vm = validity_mask if validity_mask is not None else None
                terms.append(soft_cl_dice_loss(logits[:, idx : idx + 1], target[:, idx : idx + 1], vm))
            boundary_term = torch.stack(terms).mean()
            total = total + self.cfg.boundary_weight * boundary_term

        return {
            "total": total,
            "focal": focal_per_class.mean(),
            "dice": dice_per_class.mean(),
            "boundary": boundary_term,
            "focal_per_class": focal_per_class.detach(),
            "dice_per_class": dice_per_class.detach(),
        }
