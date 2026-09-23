"""Deterministic, persisted train/val carving for datasets that ship without a labeled test set.

Used for dacl10k (user-confirmed strategy: carve a fixed-seed val subset out of `train`, promote
the official `validation` split to `test` — see CLAUDE.md). The carve is computed once and
written to disk; every subsequent run reads the persisted file rather than recomputing, so the
split never drifts across experiments (tests/test_split_determinism.py asserts this).
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def carve_train_val(image_ids: list[int], seed: int, val_fraction: float) -> dict[str, list[int]]:
    """Deterministically partitions `image_ids` into train/val given a fixed seed. Pure function
    of (image_ids, seed, val_fraction) — same inputs always produce the same output, independent
    of dict/set iteration order (image_ids is sorted before shuffling)."""
    ids = sorted(set(image_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = round(len(ids) * val_fraction)
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return {"train_ids": train_ids, "val_ids": val_ids}


def load_or_create_carve(
    persist_path: str | Path,
    image_ids: list[int],
    seed: int,
    val_fraction: float,
) -> dict[str, list[int]]:
    """Loads the persisted carve if it exists; otherwise computes and writes it. Never
    recomputes once a file is on disk — that's the whole point (fixed split across experiments)."""
    persist_path = Path(persist_path)
    if persist_path.exists():
        with open(persist_path) as f:
            return json.load(f)
    carve = carve_train_val(image_ids, seed, val_fraction)
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(persist_path, "w") as f:
        json.dump(carve, f, indent=2)
    return carve
