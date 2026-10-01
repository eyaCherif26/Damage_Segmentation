"""Validity-mask construction: which pixels should count toward loss/metrics.

Two distinct sources of invalidity, deliberately handled differently (see CLAUDE.md):

- Training data (cif_coco, dacl10k): no no-data regions in the source imagery, so the validity
  mask is all-ones. A source annotation mapped to `null` by the label mapping is NOT masked out
  here — it's a confirmed negative for the 5 MVP classes, real supervision signal, handled by
  simply not setting any mask channel (see data/common_schema.py), not by validity masking.
- Inference-time orthophotos: real no-data borders / tile seams from the 3D-model tiling process
  need to be excluded from both the model's output confidence and any downstream metrics — a
  zero-weighted-but-still-counted pixel would leak into a metric denominator.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def training_validity_mask(height: int, width: int) -> np.ndarray:
    """All-valid mask for source training/val/test imagery (no no-data regions expected)."""
    return np.ones((1, height, width), dtype=np.float32)


def no_data_validity_mask(
    image: np.ndarray,
    no_data_value: int = 0,
    tolerance: int = 3,
    min_region_px: int = 64,
    max_pixels: int | None = None,
) -> np.ndarray:
    """Detects no-data regions (borders, tile seams) in an (H, W, C) or (H, W) uint8 orthophoto.

    A pixel is a no-data *candidate* if every channel is within `tolerance` of `no_data_value`
    (tolerates slight compression/resampling noise at a hard no-data edge). Only connected
    (8-neighbour) candidate regions of at least `min_region_px` pixels are declared no-data, so an
    isolated near-black pixel or a tiny deep-shadow speck stays valid, while borders and long thin
    seam lines (length x width >= min_region_px) are excluded.

    `max_pixels`: if set and the image is larger, detection runs on a stride-subsampled copy (exact
    for a hard-zero fill, ~k^2 faster) and the mask is repeated back up to full size; `min_region_px`
    is scaled by k^2. Meant for dataset loading of very large photos; inference keeps full resolution
    so thin tile-seam lines are not missed.

    Returns a (1, H, W) float32 mask, 1 = valid, 0 = no-data.
    """
    if max_pixels is not None and image.shape[0] * image.shape[1] > max_pixels:
        k = int(np.ceil(np.sqrt(image.shape[0] * image.shape[1] / max_pixels)))
        small = no_data_validity_mask(image[::k, ::k], no_data_value, tolerance, max(4, min_region_px // (k * k)))
        full = np.repeat(np.repeat(small[0], k, axis=0), k, axis=1)[: image.shape[0], : image.shape[1]]
        return np.ascontiguousarray(full[None])

    lo, hi = max(0, no_data_value - tolerance), no_data_value + tolerance
    near = (image >= lo) & (image <= hi)
    candidate = near.all(axis=-1) if image.ndim == 3 else near

    labeled, n_regions = ndimage.label(candidate, structure=np.ones((3, 3), dtype=bool))
    if n_regions == 0:
        return np.ones((1, *candidate.shape), dtype=np.float32)
    sizes = np.bincount(labeled.ravel())
    is_no_data_region = sizes >= min_region_px
    is_no_data_region[0] = False  # label 0 is the background (non-candidate pixels)
    return (~is_no_data_region[labeled]).astype(np.float32)[None, :, :]


def apply_validity(tensor: np.ndarray, validity_mask: np.ndarray) -> np.ndarray:
    """Broadcasts a (1, H, W) validity mask across a (C, H, W) tensor's channel dim."""
    return tensor * validity_mask
