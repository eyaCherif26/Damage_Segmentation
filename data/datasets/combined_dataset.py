"""Builds train/val/test datasets across one or more sources under the unified MVP taxonomy.

Default (recommended) MVP configuration trains on the combined cif_coco + dacl10k data
(`cfg.data.sources = ["cif_coco", "dacl10k"]`); a single-source run (e.g. for a baseline-
comparison ablation against each paper's own reported numbers) is just `sources: [cif_coco]` or
`sources: [dacl10k]` — no code path difference, both go through this same function.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, Dataset

from configs.schema import MVP_CLASSES, DataConfig, DatasetSourceConfig
from data.datasets.crop_dataset import CocoCropDataset
from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset
from data.schema_validator import assert_disjoint_image_ids
from data.split_utils import load_or_create_carve

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _all_image_ids(coco_path: str) -> list[int]:
    with open(coco_path) as f:
        coco = json.load(f)
    return [img["id"] for img in coco["images"]]


def _dacl10k_split_image_ids(source_cfg: DatasetSourceConfig, data_cfg: DataConfig) -> dict[str, list[int]]:
    """Applies the user-confirmed dacl10k test strategy (see CLAUDE.md): carve a fixed-seed val
    subset out of the official train split, promote the official val split to the test role."""
    train_ids = _all_image_ids(source_cfg.annotations["train"])
    val_ids = _all_image_ids(source_cfg.annotations["val"])

    if data_cfg.dacl10k_test_strategy == "carve_val_from_train":
        persist_path = REPO_ROOT / "data" / "splits" / f"dacl10k_train_val_carve_seed{data_cfg.split_seed}.json"
        carve = load_or_create_carve(persist_path, train_ids, data_cfg.split_seed, data_cfg.dacl10k_val_fraction)
        assert_disjoint_image_ids(
            set(carve["train_ids"]), set(carve["val_ids"]), names=["dacl10k_carved_train", "dacl10k_carved_val"]
        )
        return {"train": carve["train_ids"], "val": carve["val_ids"], "test": val_ids}
    elif data_cfg.dacl10k_test_strategy == "val_only":
        return {"train": train_ids, "val": val_ids, "test": val_ids}
    else:
        raise ValueError(f"Unknown dacl10k_test_strategy: {data_cfg.dacl10k_test_strategy!r}")


def _make_dataset(
    role: str,
    images_root: str,
    coco_path: str,
    label_mapping_path: str,
    image_ids: list[int] | None,
    transform: Any,
    source_name: str,
    data_cfg: DataConfig,
    val_mode: str,
) -> Dataset:
    """One dataset for one (source, role). `whole_image` resizes every image; `native_crop` samples
    native-resolution windows: train = random crops, val = `val_mode` ("fixed" crops for monitoring or
    a full "grid" for threshold selection), test = full grid of tiles."""
    # The full fixed taxonomy union'd with this run's own (possibly reduced) mvp_classes: lets a
    # label map target a class this run excludes (treated as a confirmed negative) while still
    # catching a genuine typo (a target that's neither a canonical nor a newly-added class name).
    canonical_classes = set(MVP_CLASSES) | set(data_cfg.mvp_classes)
    if data_cfg.resize_mode == "whole_image":
        return CocoMultiLabelSegDataset(
            images_root, coco_path, label_mapping_path, data_cfg.mvp_classes, image_ids=image_ids,
            transform=transform, source_name=source_name, detect_no_data=data_cfg.no_data_validity,
            canonical_classes=canonical_classes,
        )
    mode = {"train": "random", "val": val_mode, "test": "grid"}[role]
    return CocoCropDataset(
        images_root, coco_path, label_mapping_path, data_cfg.mvp_classes, image_ids=image_ids,
        transform=transform, source_name=source_name, detect_no_data=data_cfg.no_data_validity,
        canonical_classes=canonical_classes,
        crop_size=data_cfg.image_size, scale=data_cfg.pixel_scale, mode=mode,
        scale_jitter=data_cfg.train_scale_jitter if role == "train" else 0.0,
        fg_crop_prob=data_cfg.fg_crop_prob,
        crops_per_image=data_cfg.train_crops_per_image if role == "train" else data_cfg.val_crops_per_image,
        seed=data_cfg.split_seed,
    )


def _build_source_split_datasets(
    source_name: str,
    source_cfg: DatasetSourceConfig,
    data_cfg: DataConfig,
    transform_train: Any,
    transform_eval: Any,
    val_mode: str = "fixed",
) -> dict[str, Dataset]:
    lm = source_cfg.label_mapping_path
    if source_name == "dacl10k":
        ids = _dacl10k_split_image_ids(source_cfg, data_cfg)
        return {
            "train": _make_dataset("train", source_cfg.images_roots["train"], source_cfg.annotations["train"], lm, ids["train"], transform_train, "dacl10k", data_cfg, val_mode),
            "val": _make_dataset("val", source_cfg.images_roots["train"], source_cfg.annotations["train"], lm, ids["val"], transform_eval, "dacl10k", data_cfg, val_mode),
            "test": _make_dataset("test", source_cfg.images_roots["val"], source_cfg.annotations["val"], lm, ids["test"], transform_eval, "dacl10k", data_cfg, val_mode),
        }

    # Generic path (cif_coco, and any future source with its own provided train/val/test splits)
    result = {}
    for split, transform in (("train", transform_train), ("val", transform_eval), ("test", transform_eval)):
        if split in source_cfg.annotations:
            result[split] = _make_dataset(split, source_cfg.images_roots[split], source_cfg.annotations[split], lm, None, transform, source_name, data_cfg, val_mode)
    return result


def build_split_datasets(
    data_cfg: DataConfig,
    transform_train: Any,
    transform_eval: Any,
    val_mode: str = "fixed",
) -> dict[str, Dataset]:
    """`val_mode` only matters for `resize_mode: native_crop`: "fixed" = deterministic crops (cheap,
    for per-epoch monitoring), "grid" = every tile of every val image (for unbiased threshold selection)."""
    per_split: dict[str, list[Dataset]] = {"train": [], "val": [], "test": []}
    for source_name in data_cfg.sources:
        source_cfg = data_cfg.sources_resolved[source_name]
        source_splits = _build_source_split_datasets(source_name, source_cfg, data_cfg, transform_train, transform_eval, val_mode)
        for split, ds in source_splits.items():
            per_split[split].append(ds)

    return {split: ConcatDataset(datasets) for split, datasets in per_split.items() if datasets}
