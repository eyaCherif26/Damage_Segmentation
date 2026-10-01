"""Native-resolution crop / tile datasets: train at (near-)native pixels instead of squeezing whole images.

Why: resizing a 4000x3000 photo to 512x512 throws away most of its pixels. Here the image is scaled by
`scale` (1.0 = native pixels), and the network sees `crop_size` x `crop_size` windows of it:

    mode="random"  training: a random window; with probability `fg_crop_prob` it is centred near a
                   labelled object (so crops are not mostly empty background); optional log-uniform
                   scale jitter; `crops_per_image` windows per image per epoch.
    mode="fixed"   validation monitoring: the same deterministic windows every epoch (seeded by image id),
                   so the early-stopping signal is comparable across epochs.
    mode="grid"    evaluation: a full non-overlapping grid of tiles covering every pixel of every image,
                   so metrics describe the whole image at that scale.

Labels are rasterised straight into the window (polygon coordinates are shifted and scaled), so a
24-megapixel mask is never materialised. Image and labels use one shared geometric mapping, so they
cannot drift apart. Windows that extend past the image are padded and marked invalid (validity 0), like
no-data borders. The item format is identical to `CocoMultiLabelSegDataset`.

The image is decoded with OpenCV's default (EXIF-orientation-aware) decoder and its shape is checked
against the COCO width/height, because the annotation coordinates follow that orientation.
"""
from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from data.common_schema import decode_segmentation
from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset
from data.label_mapping import LabelMapping
from data.validity_mask import no_data_validity_mask

MODES = ("random", "fixed", "grid")


class CocoCropDataset(CocoMultiLabelSegDataset):
    def __init__(
        self,
        images_root: str | Path,
        coco_path: str | Path,
        label_mapping_path: str | Path | LabelMapping,
        mvp_classes: list[str],
        image_ids: Optional[list[int]] = None,
        transform: Optional[Any] = None,
        source_name: str = "",
        detect_no_data: bool = False,
        canonical_classes: Optional[Any] = None,
        *,
        crop_size: int = 512,
        scale: float = 1.0,
        mode: str = "random",
        scale_jitter: float = 0.0,
        fg_crop_prob: float = 0.7,
        crops_per_image: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(images_root, coco_path, label_mapping_path, mvp_classes, image_ids, transform, source_name,
                          detect_no_data, canonical_classes)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if scale <= 0 or crop_size < 1 or crops_per_image < 1:
            raise ValueError("scale must be > 0, crop_size >= 1 and crops_per_image >= 1")
        self.crop_size, self.scale, self.mode = crop_size, scale, mode
        self.scale_jitter, self.fg_crop_prob = scale_jitter, fg_crop_prob
        self.crops_per_image, self.seed = crops_per_image, seed
        self._rng: np.random.Generator | None = None
        self._cache: tuple[Any, np.ndarray | None] = (None, None)

        self._tiles: list[tuple[int, int, int]] = []
        if mode == "grid":
            for pos, image_id in enumerate(self.image_ids):
                info = self.images_by_id[image_id]
                ws, hs = max(1, round(info["width"] * scale)), max(1, round(info["height"] * scale))
                for ty in range(math.ceil(hs / crop_size)):
                    for tx in range(math.ceil(ws / crop_size)):
                        self._tiles.append((pos, tx, ty))

    # ------------------------------------------------------------------ indexing
    def __len__(self) -> int:
        return len(self._tiles) if self.mode == "grid" else len(self.image_ids) * self.crops_per_image

    def sample_image_id(self, index: int) -> int:
        """The image a dataset index refers to (indices are tiles/crops, not images)."""
        if self.mode == "grid":
            return self.image_ids[self._tiles[index][0]]
        return self.image_ids[index % len(self.image_ids)]

    def _rng_for(self, index: int, image_id: Any) -> np.random.Generator:
        if self.mode == "fixed":
            return np.random.default_rng([self.seed, zlib.crc32(str(image_id).encode()), index // len(self.image_ids)])
        if self._rng is None:  # created lazily inside the worker process: distinct stream per worker and epoch
            self._rng = np.random.default_rng((torch.initial_seed() + self.seed) % 2**32)
        return self._rng

    # ------------------------------------------------------------------ geometry
    def _decode(self, image_id: int, info: dict) -> np.ndarray:
        if self._cache[0] == image_id:
            return self._cache[1]
        img = self._load_image(info["file_name"])
        if img.shape[:2] != (info["height"], info["width"]):
            raise ValueError(
                f"{info['file_name']}: decoded image is {img.shape[1]}x{img.shape[0]} but the annotations say "
                f"{info['width']}x{info['height']} — crop coordinates would be misaligned (EXIF orientation?)."
            )
        self._cache = (image_id, img)
        return img

    def _choose_origin(self, info: dict, anns: list[dict], scale: float, rng: np.random.Generator) -> tuple[float, float]:
        """Top-left corner of the window in scaled coordinates."""
        c = self.crop_size
        max_ox = max(0.0, info["width"] * scale - c)
        max_oy = max(0.0, info["height"] * scale - c)
        mapped = [a for a in anns if a["category_id"] in self.category_id_to_channel and (a.get("bbox") or a.get("segmentation"))]
        if mapped and rng.random() < self.fg_crop_prob:
            ann = mapped[int(rng.integers(len(mapped)))]
            px, py = self._point_on(ann, rng)
            cx = px * scale + rng.uniform(-0.35, 0.35) * c
            cy = py * scale + rng.uniform(-0.35, 0.35) * c
            return float(np.clip(cx - c / 2, 0, max_ox)), float(np.clip(cy - c / 2, 0, max_oy))
        return float(rng.uniform(0, max_ox)) if max_ox > 0 else 0.0, float(rng.uniform(0, max_oy)) if max_oy > 0 else 0.0

    @staticmethod
    def _point_on(ann: dict, rng: np.random.Generator) -> tuple[float, float]:
        """A native-pixel point on the object: a random vertex of one of its polygons. (The bounding-box
        centre is NOT used: for an elongated, curved or L-shaped object — cracks, rust stripes — it can lie
        far from the object, and the crop would miss it entirely.) Falls back to the bbox centre for RLE."""
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            polys = [p for p in seg if len(p) >= 6]
            if polys:
                poly = polys[int(rng.integers(len(polys)))]
                k = int(rng.integers(len(poly) // 2))
                return float(poly[2 * k]), float(poly[2 * k + 1])
        x, y, w, h = ann["bbox"]
        return x + w / 2, y + h / 2

    def render(self, image_id: int, ox: float, oy: float, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Window at scaled-coordinate origin (ox, oy). Returns (image HxWx3 uint8, masks HxWxC float32,
        pad_valid HxW float32), all crop_size x crop_size."""
        info = self.images_by_id[image_id]
        w_full, h_full = info["width"], info["height"]
        c = self.crop_size
        img = self._decode(image_id, info)

        rx0 = min(max(int(round(ox / scale)), 0), w_full - 1)
        ry0 = min(max(int(round(oy / scale)), 0), h_full - 1)
        rw_ideal, rh_ideal = max(1, int(round(c / scale))), max(1, int(round(c / scale)))
        rw, rh = min(w_full - rx0, rw_ideal), min(h_full - ry0, rh_ideal)
        ow = c if rw == rw_ideal else max(1, min(c, int(round(rw * scale))))
        oh = c if rh == rh_ideal else max(1, min(c, int(round(rh * scale))))

        roi = img[ry0 : ry0 + rh, rx0 : rx0 + rw]
        if (ow, oh) != (rw, rh):
            roi = cv2.resize(roi, (ow, oh), interpolation=cv2.INTER_AREA if (ow < rw or oh < rh) else cv2.INTER_LINEAR)
        canvas = np.zeros((c, c, 3), np.uint8)
        canvas[:oh, :ow] = roi
        pad_valid = np.zeros((c, c), np.float32)
        pad_valid[:oh, :ow] = 1.0

        sx, sy = ow / rw, oh / rh
        masks = np.zeros((c, c, len(self.mvp_classes)), np.float32)
        for ann in self.annotations_by_image.get(image_id, []):
            ch = self.category_id_to_channel.get(ann["category_id"])
            if ch is None:
                continue
            bbox = ann.get("bbox")
            if bbox and (bbox[0] > rx0 + rw or bbox[1] > ry0 + rh or bbox[0] + bbox[2] < rx0 or bbox[1] + bbox[3] < ry0):
                continue  # annotation lies entirely outside this window
            seg = ann["segmentation"]
            if isinstance(seg, list):
                polys = [
                    [((v - rx0) * sx) if k % 2 == 0 else ((v - ry0) * sy) for k, v in enumerate(poly)]
                    for poly in seg
                    if len(poly) >= 6
                ]
                if not polys:
                    continue
                m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polys, c, c))).astype(np.float32)
            else:  # RLE: no cheap window rasterisation, decode natively then crop
                m_roi = decode_segmentation(seg, h_full, w_full)[ry0 : ry0 + rh, rx0 : rx0 + rw]
                m = np.zeros((c, c), np.float32)
                m[:oh, :ow] = cv2.resize(m_roi, (ow, oh), interpolation=cv2.INTER_NEAREST)
            masks[:, :, ch] = np.maximum(masks[:, :, ch], m)
        masks *= pad_valid[:, :, None]
        return canvas, masks, pad_valid

    # ------------------------------------------------------------------ item
    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.sample_image_id(index)
        info = self.images_by_id[image_id]
        if self.mode == "grid":
            _, tx, ty = self._tiles[index]
            scale, ox, oy = self.scale, float(tx * self.crop_size), float(ty * self.crop_size)
        else:
            rng = self._rng_for(index, image_id)
            scale = self.scale
            if self.mode == "random" and self.scale_jitter > 0:
                span = math.log1p(self.scale_jitter)
                scale = self.scale * math.exp(rng.uniform(-span, span))
            ox, oy = self._choose_origin(info, self.annotations_by_image.get(image_id, []), scale, rng)

        image, masks, validity = self.render(image_id, ox, oy, scale)
        if self.detect_no_data:
            validity = validity * no_data_validity_mask(image)[0]
        return self._finalize(image, masks, validity, image_id)
