"""Sliding-window inference over large orthophotos, generalized from a sibling project's
`SegF_model/orthophoto_inference.py` (the closest existing prior art — true probability-map
averaging via float32 accumulators, not OR/union) to work with ANY registered model, not just
SegFormer, via the uniform `forward(pixel_values) -> (B, C, H, W)` logits contract every
`models/registry.py` entry satisfies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from data.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from data.validity_mask import no_data_validity_mask
from inference.vram_budget import check_budget


@dataclass
class TileOrigin:
    x0: int
    y0: int


def compute_tile_origins(width: int, height: int, tile_size: int, overlap: int) -> list[TileOrigin]:
    """Covers the full image with `stride = tile_size - overlap`, guaranteeing the last row/column
    is flush with the image edge (extra overlap there rather than a partial out-of-bounds tile)."""
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    return [TileOrigin(x, y) for y in ys for x in xs]


@torch.no_grad()
def run_tiled_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    num_classes: int,
    tile_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    max_memory_gb: float = 8.0,
    allow_large_memory: bool = False,
    no_data_value: int = 0,
) -> np.ndarray:
    """image: (H, W, 3) uint8 RGB. Returns (num_classes, H, W) float32 probability map, blended
    via true averaging over overlapping tile predictions (accumulate prob_sum + count, divide
    once at the end) and gated by a no-data validity mask so black orthophoto borders/tile seams
    never contribute spurious detections.
    """
    height, width = image.shape[:2]
    check_budget(num_classes, width, height, max_memory_gb, allow_large_memory)

    validity = no_data_validity_mask(image, no_data_value=no_data_value)[0]  # (H, W)

    prob_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    count = np.zeros((1, height, width), dtype=np.float32)

    origins = compute_tile_origins(width, height, tile_size, overlap)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    model.eval()
    for batch_start in range(0, len(origins), batch_size):
        batch_origins = origins[batch_start : batch_start + batch_size]
        tiles = []
        for origin in batch_origins:
            tile = image[origin.y0 : origin.y0 + tile_size, origin.x0 : origin.x0 + tile_size]
            th, tw = tile.shape[:2]
            if (th, tw) != (tile_size, tile_size):
                padded = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
                padded[:th, :tw] = tile
                tile = padded
            normalized = (tile.astype(np.float32) / 255.0 - mean) / std
            tiles.append(normalized.transpose(2, 0, 1))

        batch_tensor = torch.from_numpy(np.stack(tiles)).float().to(device)
        logits = model(batch_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i, origin in enumerate(batch_origins):
            th = min(tile_size, height - origin.y0)
            tw = min(tile_size, width - origin.x0)
            prob_sum[:, origin.y0 : origin.y0 + th, origin.x0 : origin.x0 + tw] += probs[i, :, :th, :tw]
            count[:, origin.y0 : origin.y0 + th, origin.x0 : origin.x0 + tw] += 1.0

    count = np.maximum(count, 1e-6)
    prob_map = prob_sum / count
    prob_map = prob_map * validity[None, :, :]
    return prob_map
