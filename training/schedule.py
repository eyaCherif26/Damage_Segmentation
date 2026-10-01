"""Cosine LR schedule with linear warmup, matching the dacl10k baseline recipe `[Literature]`.

Built once, at optimizer construction, against param groups that already both exist (backbone +
head) — see training/trainer.py. This project never calls `optimizer.add_param_group()`
mid-training specifically to avoid the scheduler-desync bug documented in CLAUDE.md (a group
added after `LambdaLR`/`_LRScheduler` snapshots `base_lrs` at construction silently never gets
updated again).
"""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    steps_per_epoch: int,
    min_lr: float = 1e-5,
) -> LambdaLR:
    """`min_lr` is an ABSOLUTE floor shared by every param group (not a fraction of that group's own
    peak LR), applied only to the post-warmup cosine phase — warmup itself still ramps from exactly
    0 regardless of `min_lr`, matching the literal "don't let the cosine ANNEALING go below this"
    ask (see CLAUDE.md, 2026-09-29). `LambdaLR` applies one callable per param group (a list, not a
    single shared one), because each group's own base LR (e.g. lr_backbone vs. lr_head) needs a
    different multiplier to reach the SAME absolute floor.
    """
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, max_epochs * steps_per_epoch)

    def make_lr_lambda(base_lr: float):
        min_mult = (min_lr / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(cosine_mult, min_mult)

        return lr_lambda

    lr_lambdas = [make_lr_lambda(group["lr"]) for group in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambdas)
