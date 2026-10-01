import json

import albumentations as A
import cv2
import numpy as np
import pytest
import yaml

from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset

MVP = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]
H, W = 120, 160


@pytest.fixture
def tiny_coco(tmp_path):
    img = np.full((H, W, 3), 140, np.uint8)
    img[:, :40] = 0  # black no-data padding on the left
    cv2.imwrite(str(tmp_path / "a.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    img[:, 130:, 0] = 250  # a red strip on the right edge, to check RGB order survives loading
    cv2.imwrite(str(tmp_path / "b.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    square = lambda x0, y0, x1, y1: [[x0, y0, x1, y0, x1, y1, x0, y1]]
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": W, "height": H}, {"id": 2, "file_name": "b.jpg", "width": W, "height": H}],
        "categories": [{"id": 1, "name": "Crack"}, {"id": 2, "name": "Rust"}, {"id": 3, "name": "Graffiti"}, {"id": 4, "name": "Spalling"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 4, "segmentation": square(60, 20, 120, 100), "area": 4800, "bbox": [60, 20, 60, 80], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": square(80, 40, 100, 60), "area": 400, "bbox": [80, 40, 20, 20], "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 3, "segmentation": square(45, 5, 55, 15), "area": 100, "bbox": [45, 5, 10, 10], "iscrowd": 0},
        ],
    }
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    (tmp_path / "map.yaml").write_text(yaml.safe_dump({"Crack": "Cracks", "Rust": "Rust", "Spalling": "Spalling", "Graffiti": None}))
    return tmp_path


def _ds(root, **kw):
    return CocoMultiLabelSegDataset(root, root / "coco.json", root / "map.yaml", MVP, **kw)


def test_item_shapes_channel_order_and_multilabel_overlap(tiny_coco):
    item = _ds(tiny_coco)[0]
    assert item["image"].shape == (3, H, W) and item["mask"].shape == (5, H, W) and item["validity_mask"].shape == (1, H, W)
    m = item["mask"]
    assert m[2].sum() > 0 and m[3].sum() > 0 and m[0].sum() == 0 and m[1].sum() == 0 and m[4].sum() == 0
    assert ((m[2] > 0) & (m[3] > 0)).sum() > 300, "rust inside spalling must appear in both channels"
    assert item["source_dataset"] == "coco"


def test_null_mapped_class_adds_nothing_and_is_not_invalidated(tiny_coco):
    item = _ds(tiny_coco, detect_no_data=True)[0]
    assert item["mask"].sum(dim=(1, 2))[[0, 1, 4]].sum() == 0
    assert item["validity_mask"][0, 5:15, 45:55].min() == 1, "a Graffiti (null-mapped) polygon is a confirmed negative, not an ignore region"


def test_no_data_padding_is_invalid_only_when_enabled(tiny_coco):
    on = _ds(tiny_coco, detect_no_data=True)[0]["validity_mask"]
    off = _ds(tiny_coco, detect_no_data=False)[0]["validity_mask"]
    assert on[0, :, :35].max() == 0 and on[0, :, 45:].min() == 1
    assert off.min() == 1


def test_image_is_rgb_not_bgr(tiny_coco):
    img = _ds(tiny_coco)[1]["image"]  # no transform -> values in [0, 1]
    assert img[0, :, 140:].mean() > 0.9 and img[2, :, 140:].mean() < 0.65, "red strip must be in channel 0"


def test_geometric_transform_keeps_image_mask_and_validity_aligned(tiny_coco):
    flip = A.Compose([A.HorizontalFlip(p=1.0), A.Resize(64, 96)])
    plain = _ds(tiny_coco, detect_no_data=True)[0]
    flipped = _ds(tiny_coco, detect_no_data=True, transform=flip)[0]
    assert flipped["mask"].shape == (5, 64, 96)
    assert flipped["validity_mask"][0, :, -10:].max() == 0, "padding moved from the left edge to the right edge"
    assert plain["mask"][2, :, 60:120].sum() > 0 and plain["mask"][2, :, :55].sum() == 0
    # x in [60,120] of W=160 mirrors to [40,100], then Resize to width 96 -> [24,60]
    cols = flipped["mask"][2].sum(dim=0)
    assert cols[:20].sum() == 0 and cols[66:].sum() == 0 and cols[26:58].min() > 0
    assert abs(plain["mask"][2].mean() - flipped["mask"][2].mean()) < 0.03


def test_image_id_subset_and_missing_id_error(tiny_coco):
    assert len(_ds(tiny_coco, image_ids=[2])) == 1
    with pytest.raises(KeyError):
        _ds(tiny_coco, image_ids=[99])


def test_label_mapping_missing_category_fails_loudly(tiny_coco):
    (tiny_coco / "map.yaml").write_text(yaml.safe_dump({"Crack": "Cracks"}))
    with pytest.raises(KeyError):
        _ds(tiny_coco)


def test_reduced_mvp_classes_excludes_unselected_class_instead_of_raising(tiny_coco):
    """A label map written against the full taxonomy (here: Spalling included) must still work,
    unmodified, when a run selects a reduced mvp_classes subset that leaves that class out —
    'Spalling' is a real class, just not selected for this run, so its annotation is silently
    excluded (like a null mapping), not an error. canonical_classes carries the full taxonomy
    separately, so a genuine typo is still caught (next test)."""
    ds = CocoMultiLabelSegDataset(
        tiny_coco, tiny_coco / "coco.json", tiny_coco / "map.yaml", ["Cracks", "Rust"],
        canonical_classes=MVP,
    )
    item = ds[0]
    assert item["mask"].shape == (2, H, W)
    assert item["mask"][1].sum() > 0, "Rust annotation still present"
    # Spalling's annotation contributed to no channel at all (dataset only has 2 channels: Cracks, Rust)


def test_canonical_classes_catches_typo_even_with_a_reduced_mvp_classes_list(tiny_coco):
    (tiny_coco / "map.yaml").write_text(yaml.safe_dump(
        {"Crack": "Cracks", "Rust": "Rust", "Spalling": "Spalllng", "Graffiti": None}  # typo: 'Spalllng'
    ))
    with pytest.raises(ValueError, match="Spalllng"):
        CocoMultiLabelSegDataset(
            tiny_coco, tiny_coco / "coco.json", tiny_coco / "map.yaml", ["Cracks", "Rust"],
            canonical_classes=MVP,
        )


def test_train_transform_includes_shadow_and_never_touches_masks_or_validity():
    from configs.schema import AugmentationConfig
    from data.datasets.transforms import build_train_transform

    aug = AugmentationConfig(horizontal_flip_p=0, vertical_flip_p=0, rotate90_p=0, shift_scale_rotate_p=0, blur_p=0, color_jitter_p=0, shadow_p=1.0)
    t = build_train_transform(aug, 64)
    img = np.full((64, 64, 3), 200, np.uint8)
    mask = np.zeros((64, 64, 6), np.float32)
    mask[10:20, 10:20, 2] = 1
    out = t(image=img, mask=mask)
    assert np.array_equal(out["mask"], mask), "photometric augmentation must not change label/validity channels"
    ref = build_train_transform(AugmentationConfig(horizontal_flip_p=0, vertical_flip_p=0, rotate90_p=0, shift_scale_rotate_p=0, blur_p=0, color_jitter_p=0, shadow_p=0), 64)(image=img, mask=mask)["image"]
    assert not np.allclose(out["image"], ref), "a shadow must actually darken part of the image"
