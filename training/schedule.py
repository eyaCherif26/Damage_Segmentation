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
) -> LambdaLR:
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, max_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
