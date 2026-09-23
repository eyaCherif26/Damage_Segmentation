"""Unlabelled images from a folder (new data without ground truth), in the same item format as
`CocoMultiLabelSegDataset` so the same model/visualisation code path is used.

Images are resized to the evaluation resolution by the shared eval transform — exactly what the model
saw at validation time. No-data (black padding) is detected the same way as for the labelled datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.validity_mask import no_data_validity_mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(root: str | Path, recursive: bool = False) -> list[Path]:
    """Image files under `root`, sorted. Matches on the *final* suffix, so Windows metadata files such as
    `photo.jpg:Zone.Identifier` (suffix `.Identifier`) are ignored."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}")
    files = root.rglob("*") if recursive else root.glob("*")
    return sorted(f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


class ImageFolderDataset(Dataset):
    has_ground_truth = False

    def __init__(
        self,
        root: str | Path,
        num_classes: int,
        transform: Optional[Any] = None,
        recursive: bool = False,
        detect_no_data: bool = True,
        source_name: str = "",
    ) -> None:
        self.root = Path(root)
        self.files = list_images(self.root, recursive)
        if not self.files:
            raise FileNotFoundError(f"No images ({sorted(IMAGE_EXTENSIONS)}) found in {self.root}")
        self.num_classes = num_classes
        self.transform = transform
        self.detect_no_data = detect_no_data
        self.source_name = source_name or self.root.name

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        validity = no_data_validity_mask(image, max_pixels=2_000_000)[0] if self.detect_no_data else np.ones((h, w), np.float32)
        combined = np.concatenate([np.zeros((h, w, self.num_classes), np.float32), validity[:, :, None]], axis=-1)

        if self.transform is not None:
            out = self.transform(image=image, mask=combined)
            image_out, combined_out = out["image"], out["mask"]
        else:
            image_out, combined_out = image.astype(np.float32) / 255.0, combined

        def to_tensor(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (2, 0, 1)))).float()

        return {
            "image": to_tensor(image_out),
            "mask": to_tensor(combined_out[:, :, :-1]),
            "validity_mask": to_tensor(combined_out[:, :, -1:]),
            "image_id": str(path.relative_to(self.root).with_suffix("")).replace("/", "__"),
            "source_dataset": self.source_name,
        }
