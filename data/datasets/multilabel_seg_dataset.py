"""Generic multi-label COCO segmentation dataset.

One class, not per-source subclasses: once cif_coco and dacl10k are both plain COCO files with a
label-mapping YAML applied, there is no dataset-specific logic left — pointing this at a
different dataset later is a config change (new configs/data/datasets/*.yaml +
configs/data/label_maps/*.yaml), not a code change.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.common_schema import rasterize_annotations
from data.label_mapping import LabelMapping
from data.validity_mask import no_data_validity_mask, training_validity_mask


class CocoMultiLabelSegDataset(Dataset):
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
        canonical_classes: Optional[Collection[str]] = None,
    ) -> None:
        """`canonical_classes`, if given, is the full class taxonomy this dataset's label map is
        checked against (independent of `mvp_classes`, this run's *selected* subset) — catches a
        typo'd label-map target before it can be silently mistaken for a deliberate exclusion.
        Callers training on a reduced `mvp_classes` should pass the union of the fixed taxonomy
        and this run's own `mvp_classes` (see data/datasets/combined_dataset.py)."""
        self.images_root = Path(images_root)
        self.source_name = source_name or Path(coco_path).stem
        self.mvp_classes = list(mvp_classes)
        self.transform = transform
        self.detect_no_data = detect_no_data

        with open(coco_path) as f:
            coco = json.load(f)

        self.images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}
        category_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

        label_mapping = label_mapping_path if isinstance(label_mapping_path, LabelMapping) else LabelMapping.load(label_mapping_path)
        if canonical_classes is not None:
            label_mapping.validate_targets(canonical_classes)
        label_mapping.check_covers(list(category_id_to_name.values()))
        self.category_id_to_channel = label_mapping.build_category_id_to_channel(category_id_to_name, self.mvp_classes)

        self.annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations_by_image[ann["image_id"]].append(ann)

        available_ids = set(self.images_by_id)
        if image_ids is not None:
            missing = set(image_ids) - available_ids
            if missing:
                raise KeyError(f"{self.source_name}: requested image_ids not in the COCO file: {sorted(missing)[:10]}...")
            self.image_ids = sorted(image_ids)
        else:
            self.image_ids = sorted(available_ids)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, file_name: str) -> np.ndarray:
        path = self.images_root / file_name
        # Default decode on purpose: it applies the EXIF orientation, which the annotation coordinates
        # follow (verified: 10/10 rotated dacl10k train images and 1/1 test image match COCO dims only
        # this way). cv2.IMREAD_IGNORE_ORIENTATION and the fast IMREAD_REDUCED_* modes would misalign labels.
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # explicit BGR->RGB; see CLAUDE.md's channel-order note

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        info = self.images_by_id[image_id]
        image = self._load_image(info["file_name"])
        height, width = info["height"], info["width"]

        anns = self.annotations_by_image.get(image_id, [])
        mask = rasterize_annotations(anns, height, width, self.category_id_to_channel, len(self.mvp_classes))
        mask_hwc = np.transpose(mask, (1, 2, 0))  # C,H,W -> H,W,C for albumentations
        if self.detect_no_data:
            validity = no_data_validity_mask(image, max_pixels=2_000_000)[0]  # (H, W); black padding is excluded from loss/metrics
        else:
            validity = training_validity_mask(height, width)[0]

        return self._finalize(image, mask_hwc, validity, image_id)

    def _finalize(self, image: np.ndarray, mask_hwc: np.ndarray, validity: np.ndarray, image_id: Any) -> dict[str, Any]:
        """Applies the transform to image + (masks, validity) together and converts to tensors.
        Shared with the crop datasets so both produce byte-identical item formats."""
        combined_mask = np.concatenate([mask_hwc, validity[:, :, None]], axis=-1)

        if self.transform is not None:
            out = self.transform(image=image, mask=combined_mask)
            image_out, combined_out = out["image"], out["mask"]
        else:
            image_out, combined_out = image.astype(np.float32) / 255.0, combined_mask

        mask_out = combined_out[:, :, :-1]
        validity_out = combined_out[:, :, -1:]

        image_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(image_out, (2, 0, 1)))).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(mask_out, (2, 0, 1)))).float()
        validity_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(validity_out, (2, 0, 1)))).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "validity_mask": validity_tensor,
            "image_id": image_id,
            "source_dataset": self.source_name,
        }
