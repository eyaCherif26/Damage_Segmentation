"""Shared internal record contract: COCO annotations -> multi-channel multi-label mask.

Every dataset loader in `data/datasets/` normalizes into the same per-sample dict:

    {
      "image": FloatTensor[3, H, W],          # normalized, augmented if train
      "mask": FloatTensor[C, H, W],           # multi-label binary, C = len(mvp_classes)
      "validity_mask": FloatTensor[1, H, W],  # 1 = supervise this pixel, 0 = ignore
      "image_id": int,
      "source_dataset": str,
    }

Rasterization happens on-the-fly in `__getitem__` (not pre-baked to disk): both source datasets
are COCO-format at a size pycocotools decodes fast, and on-the-fly keeps the label-mapping config
hot-swappable without a re-rasterization pass over the whole dataset.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def decode_segmentation(segmentation: list | dict, height: int, width: int) -> np.ndarray:
    """COCO `segmentation` (polygon list, uncompressed RLE dict, or compressed RLE dict) -> a
    single (H, W) float32 binary mask in {0, 1}. Multiple polygons for one annotation (an
    annotation with holes/disjoint parts) are merged via OR, matching COCO's own semantics for
    a single instance's segmentation field."""
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return np.zeros((height, width), dtype=np.float32)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.float32)


def rasterize_annotations(
    annotations: list[dict],
    height: int,
    width: int,
    category_id_to_channel: dict[int, int],
    num_channels: int,
) -> np.ndarray:
    """List of COCO annotation dicts (each with `category_id` + `segmentation`) -> a
    (num_channels, H, W) float32 multi-label binary mask.

    Multi-label overlap semantics: each channel is built independently via per-pixel OR across
    every annotation mapped to it (`mask[ch] = maximum(mask[ch], decoded)`), so a pixel that is
    e.g. both Rust and Spalling correctly ends up 1 in both channels — no softmax/argmax anywhere
    in this function, and it must stay that way (see tests/test_mask_rasterization.py).

    Annotations whose `category_id` is not in `category_id_to_channel` (i.e. mapped to null by
    the label mapping) are skipped — they contribute to no channel, which is the correct "not one
    of the 5 MVP classes here" signal (see data/label_mapping.py and CLAUDE.md's note on why this
    is deliberately distinct from validity masking).
    """
    mask = np.zeros((num_channels, height, width), dtype=np.float32)
    for ann in annotations:
        channel = category_id_to_channel.get(ann["category_id"])
        if channel is None:
            continue
        decoded = decode_segmentation(ann["segmentation"], height, width)
        mask[channel] = np.maximum(mask[channel], decoded)
    return mask
