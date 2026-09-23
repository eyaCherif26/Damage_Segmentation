"""Patience-based early stopping on a val metric (default: val macro Dice). `max_epochs` remains
a hard ceiling regardless (see CLAUDE.md) — this only allows stopping *earlier*."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 1e-4
    mode: str = "max"  # "max" for metrics like Dice/IoU, "min" for loss

    def __post_init__(self) -> None:
        self.best: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Returns True if `value` is a new best (caller should checkpoint on True)."""
        if self.best is None:
            is_better = True
        elif self.mode == "max":
            is_better = value > self.best + self.min_delta
        else:
            is_better = value < self.best - self.min_delta

        if is_better:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return is_better
