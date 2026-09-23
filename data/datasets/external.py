"""Any labelled COCO dataset (annotation file + image folder) as an evaluation/visualisation dataset.

Used by scripts/evaluate.py and scripts/visualize_predictions.py. Nothing here knows about a specific
dataset layout: the caller passes the two paths (and optionally a label-map YAML).

  * No label map -> the file's categories must already BE MVP class names (e.g. the 5-class remapped files);
    any other category name raises instead of being dropped silently.
  * With a label map -> every category of the file needs an entry (null = confirmed negative), see
    `LabelMapping`.
  * `native_crop` models get the same treatment as the configured test split: a full non-overlapping tile
    grid at the training `pixel_scale`, not a whole-image squeeze they were never trained on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from configs.schema import DataConfig
from data.datasets.crop_dataset import CocoCropDataset
from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset
from data.label_mapping import LabelMapping


def build_labelled_dataset(
    coco_path: str | Path,
    images_root: str | Path,
    data_cfg: DataConfig,
    transform: Any,
    label_map: str | Path | None = None,
) -> tuple[Dataset, dict[str, Any]]:
    """Returns (dataset, info). `info` = {"name", "n_images", "categories", "unmapped", "absent_classes"}:
    the dataset name (used as the per-source label in result tables), the categories the file declares,
    those the mapping sends to null, and the MVP classes that no category of the file maps to."""
    coco_path, images_root = Path(coco_path), Path(images_root)
    if not coco_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"image folder not found: {images_root}")
    classes = data_cfg.mvp_classes
    with open(coco_path) as f:
        categories = [c["name"] for c in json.load(f)["categories"]]
    mapping = LabelMapping.load(label_map) if label_map else LabelMapping.identity_for(categories, classes)

    common = dict(transform=transform, source_name=coco_path.parent.name or coco_path.stem, detect_no_data=data_cfg.no_data_validity)
    if data_cfg.resize_mode == "native_crop":
        ds = CocoCropDataset(images_root, coco_path, mapping, classes, crop_size=data_cfg.image_size,
                             scale=data_cfg.pixel_scale, mode="grid", **common)
    else:
        ds = CocoMultiLabelSegDataset(images_root, coco_path, mapping, classes, **common)

    mapped_to = {mapping.resolve(c) for c in categories}
    info = {
        "name": common["source_name"],
        "n_images": len(ds.image_ids),
        "categories": categories,
        "unmapped": [c for c in categories if mapping.resolve(c) is None or mapping.resolve(c) not in classes],
        "absent_classes": [c for c in classes if c not in mapped_to],
    }
    return ds, info
