import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset

from inference.visualize import (
    INVALID_OUTLINE,
    candidate_indices,
    class_colors,
    denormalize,
    macro_dice_over_gt_classes,
    overlay,
    per_class_dice,
    render_comparison,
    stack_vertically,
)
from metrics.evaluation import load_thresholds, parse_threshold_overrides

CLASSES = ["Cracks", "Wetspot", "Spalling"]
H = W = 40


def _masks():
    gt = np.zeros((3, H, W), bool)
    gt[0, 5:15, 5:15] = True
    gt[2, 20:35, 20:35] = True
    return gt


def _image():
    return np.full((H, W, 3), 128, np.uint8)


def test_per_class_dice_values_and_nan_for_absent_classes():
    gt = _masks()
    pred = gt.copy()
    d = per_class_dice(pred, gt)
    assert d[0] == pytest.approx(1.0) and d[2] == pytest.approx(1.0) and np.isnan(d[1]), "empty-vs-empty must be NaN, not 1.0"
    half = gt.copy()
    half[0, 5:10, 5:15] = False
    assert per_class_dice(half, gt)[0] == pytest.approx(2 * 50 / (50 + 100))
    fp_only = np.zeros_like(gt)
    fp_only[1, 0:5, 0:5] = True
    assert per_class_dice(fp_only, gt)[1] == 0.0


def test_validity_mask_excludes_pixels_from_dice():
    gt = _masks()
    pred = np.zeros_like(gt)
    valid = np.ones((H, W), bool)
    valid[:, :20] = False  # hides the whole Cracks region
    d = per_class_dice(pred, gt, valid)
    assert np.isnan(d[0]), "a class whose GT is entirely invalid and which is not predicted is absent"
    assert d[2] == 0.0


def test_macro_ranking_ignores_classes_absent_from_gt():
    gt = _masks()
    pred = gt.copy()
    pred[1, 0:10, 0:10] = True  # false positive on a class with no GT
    d = per_class_dice(pred, gt)
    assert d[1] == 0.0
    assert macro_dice_over_gt_classes(d, gt) == pytest.approx(1.0), "FPs on absent classes must not affect the ranking"
    assert np.isnan(macro_dice_over_gt_classes(d, np.zeros_like(gt)))


def test_overlay_colours_class_and_keeps_pixels_outside_untouched():
    gt = _masks()
    img = _image()
    colors = class_colors(3)
    out = overlay(img, gt, colors)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[30:, :10], img[30:, :10]), "pixels outside every mask stay unchanged"
    inside = out[10, 10]
    assert inside[0] > img[10, 10][0], "Cracks (red) tints its region toward red"


def test_overlapping_classes_both_remain_visible():
    masks = np.zeros((2, H, W), bool)
    masks[0, 5:30, 5:30] = True
    masks[1, 10:25, 10:25] = True  # fully inside class 0
    out = overlay(_image(), masks, class_colors(2))
    only_a, both = out[7, 7].astype(int), out[15, 15].astype(int)
    assert not np.array_equal(only_a, both), "the overlap region must look different from single-class regions"


def test_no_data_is_tinted_and_outlined():
    img = _image()
    valid = np.ones((H, W), bool)
    valid[:, :10] = False
    out = overlay(img, np.zeros((3, H, W), bool), class_colors(3), valid)
    assert out[20, 3].astype(int).sum() < img[20, 3].astype(int).sum(), "invalid region is darkened"
    assert (out == np.array(INVALID_OUTLINE, np.uint8)).all(-1).any(), "and outlined in cyan"
    assert np.array_equal(out[20, 30], img[20, 30])


def test_masks_are_not_drawn_over_invalid_pixels():
    masks = np.zeros((1, H, W), bool)
    masks[0, :, :] = True
    valid = np.ones((H, W), bool)
    valid[:, :20] = False
    with_valid = overlay(_image(), masks, class_colors(1), valid)
    assert with_valid[20, 5].astype(int).sum() < 300 and abs(int(with_valid[20, 5][0]) - int(with_valid[20, 5][2])) < 60, "no class colour in the no-data area"


def test_denormalize_inverts_normalisation():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    norm = ((raw / 255.0 - np.asarray(mean)) / np.asarray(std)).astype(np.float32).transpose(2, 0, 1)
    assert np.abs(denormalize(norm, mean, std).astype(int) - raw.astype(int)).max() <= 1


@pytest.mark.parametrize("n_models", [1, 3])
def test_render_comparison_builds_one_column_per_model(n_models):
    gt = _masks()
    preds = {f"model_{i}": gt.copy() for i in range(n_models)}
    valid = np.ones((H, W), bool)
    valid[:, :5] = False
    fig = render_comparison(_image(), gt, preds, valid, CLASSES, "title", dpi=40)
    axes = [a for a in fig.axes if a.images]
    assert len(axes) == 2 + n_models
    assert all(a.images[0].get_array().shape == (H, W, 3) for a in axes)
    titles = [a.get_title() for a in axes]
    assert titles[0] == "Image" and titles[1] == "Ground truth" and titles[2].startswith("model_0")
    assert "macro Dice 1.00" in titles[2]
    fig.canvas.draw()
    plt.close(fig)


def test_render_handles_image_without_any_target_class():
    fig = render_comparison(_image(), np.zeros((3, H, W), bool), {"m": np.zeros((3, H, W), bool)}, None, CLASSES, "t", dpi=40)
    assert "macro" not in fig.axes[2].get_title(), "no GT classes -> no macro score is invented"
    plt.close(fig)


def test_stack_vertically_pads_to_common_width():
    a, b = np.zeros((10, 30, 3), np.uint8), np.zeros((12, 50, 3), np.uint8)
    out = stack_vertically([a, b], pad=2)
    assert out.shape == (10 + 2 + 12 + 2, 50, 3)
    assert (out[:10, 30:] == 255).all(), "the narrower image is padded with white"


class _FakeSub:
    def __init__(self, source, image_ids, ann_by_image, cat_to_ch):
        self.source_name, self.image_ids = source, image_ids
        self.annotations_by_image, self.category_id_to_channel = ann_by_image, cat_to_ch

    def __len__(self):
        return len(self.image_ids)


def _fake_concat():
    ch = {1: 0, 2: 1}  # category id -> channel; category 3 is mapped to null
    a = _FakeSub("cif", [10, 11, 12], {10: [{"category_id": 1}], 11: [{"category_id": 2}], 12: [{"category_id": 3}]}, ch)
    b = _FakeSub("dacl", [20, 21], {20: [{"category_id": 1}, {"category_id": 2}], 21: []}, ch)
    concat = type("C", (), {})()
    concat.datasets = [a, b]
    return concat


def test_candidate_indices_filters_by_source_and_class_without_reading_images():
    ds = _fake_concat()
    assert candidate_indices(ds, CLASSES) == [0, 1, 2, 3, 4]
    assert candidate_indices(ds, CLASSES, source="dacl") == [3, 4]
    assert candidate_indices(ds, CLASSES, require_classes=["Cracks"]) == [0, 3]
    assert candidate_indices(ds, CLASSES, require_classes=["Cracks", "Wetspot"]) == [3], "must contain ALL required classes"
    assert candidate_indices(ds, CLASSES, source="cif", require_classes=["Wetspot"]) == [1]
    assert candidate_indices(ds, CLASSES, source="nope") == []


def test_indices_line_up_with_a_real_concatdataset():
    class Sub(torch.utils.data.Dataset):
        def __init__(self, source, n):
            self.source_name, self.image_ids = source, list(range(n))
            self.annotations_by_image, self.category_id_to_channel = {}, {}

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, i):
            return (self.source_name, i)

    ds = ConcatDataset([Sub("a", 3), Sub("b", 4)])
    for idx in candidate_indices(ds, CLASSES, source="b"):
        assert ds[idx][0] == "b"


def test_load_thresholds(tmp_path):
    f = tmp_path / "thresholds.json"
    assert load_thresholds(f, CLASSES) is None
    f.write_text(json.dumps({c: {"threshold": 0.3 + 0.1 * i, "val_dice": 0.5} for i, c in enumerate(CLASSES)}))
    assert load_thresholds(f, CLASSES) == pytest.approx({"Cracks": 0.3, "Wetspot": 0.4, "Spalling": 0.5})
    f.write_text(json.dumps({"Cracks": {"threshold": 0.3}}))
    with pytest.raises(KeyError, match="Wetspot"):
        load_thresholds(f, CLASSES)


def test_parse_threshold_overrides():
    assert parse_threshold_overrides(["Cracks=0.3", "Wetspot=0.75"], CLASSES) == {"Cracks": 0.3, "Wetspot": 0.75}
    assert parse_threshold_overrides([], CLASSES) == {}
    with pytest.raises(ValueError, match="not one of"):
        parse_threshold_overrides(["NotAClass=0.5"], CLASSES)
    with pytest.raises(ValueError, match="not a number"):
        parse_threshold_overrides(["Cracks=oops"], CLASSES)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        parse_threshold_overrides(["Cracks=1.5"], CLASSES)
    with pytest.raises(ValueError, match="CLASS=VALUE"):
        parse_threshold_overrides(["Cracks:0.3"], CLASSES)


def test_overlay_accepts_the_memory_layout_real_images_have():
    """Regression: `denormalize` yields an image derived from a transposed CHW tensor. Its strides are
    channel-first, which cv2.drawContours rejects ('Layout of the output array image is incompatible');
    np.full-built test images hid this. Use the real code path and a deliberately awkward layout."""
    rng = np.random.default_rng(0)
    chw = rng.standard_normal((3, H, W)).astype(np.float32)
    real = denormalize(chw, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert real.flags["C_CONTIGUOUS"], "denormalize must return a contiguous array"

    awkward = np.ascontiguousarray(np.zeros((3, H, W), np.uint8)).transpose(1, 2, 0) + 100  # channel-first strides
    for image in (real, awkward, np.asfortranarray(real)):
        gt = _masks()
        out = overlay(image, gt, class_colors(3))
        assert out.shape == (H, W, 3)
        valid = np.ones((H, W), bool)
        valid[:, :5] = False
        assert overlay(image, gt, class_colors(3), valid).shape == (H, W, 3)
        fig = render_comparison(image, gt, {"m": gt}, valid, CLASSES, "t", dpi=30)
        plt.close(fig)


# ---------------------------------------------------------------- randomness -----------------------------------
from inference.visualize import pick_random, resolve_seed


def test_fresh_seed_differs_between_calls_but_given_seed_is_kept():
    seeds = {resolve_seed(None) for _ in range(20)}
    assert len(seeds) > 15, "no seed given -> every run must get a different random seed"
    assert resolve_seed(123) == 123


def test_pick_random_is_reproducible_for_a_seed_and_varies_across_seeds():
    pool = list(range(500))
    assert pick_random(pool, 12, seed=7) == pick_random(pool, 12, seed=7)
    assert pick_random(pool, 12, seed=7) != pick_random(pool, 12, seed=8)
    picks = pick_random(pool, 12, seed=7)
    assert len(set(picks)) == 12 and all(p in pool for p in picks), "distinct members of the pool"
    assert sorted(pick_random([3, 4], 10, seed=1)) == [3, 4], "asking for more than exist returns them all"


def test_two_unseeded_runs_show_different_images():
    pool = list(range(1000))
    a = pick_random(pool, 10, resolve_seed(None))
    b = pick_random(pool, 10, resolve_seed(None))
    assert a != b


# -------------------------------------------------- new data: image folders, arbitrary COCO ---------------------
import cv2

from data.datasets.image_folder_dataset import ImageFolderDataset, list_images
from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset
from data.datasets.transforms import build_eval_transform
from data.label_mapping import LabelMapping

MVP = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]


def _write_jpg(path, h=60, w=80, black_left=0):
    img = np.full((h, w, 3), 140, np.uint8)
    if black_left:
        img[:, :black_left] = 0
    cv2.imwrite(str(path), img)


def test_list_images_ignores_windows_metadata_and_non_images(tmp_path):
    _write_jpg(tmp_path / "a.jpg")
    _write_jpg(tmp_path / "B.PNG")
    (tmp_path / "a.jpg:Zone.Identifier").write_text("[ZoneTransfer]")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    _write_jpg(tmp_path / "sub" / "c.jpg")
    assert [f.name for f in list_images(tmp_path)] == ["B.PNG", "a.jpg"]
    assert [f.name for f in list_images(tmp_path, recursive=True)] == ["B.PNG", "a.jpg", "c.jpg"]
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing")


def test_image_folder_dataset_items_match_the_labelled_format(tmp_path):
    _write_jpg(tmp_path / "x.jpg", black_left=25)
    ds = ImageFolderDataset(tmp_path, len(MVP), transform=build_eval_transform(512))
    item = ds[0]
    assert item["image"].shape == (3, 512, 512) and item["mask"].shape == (5, 512, 512) and item["validity_mask"].shape == (1, 512, 512)
    assert item["mask"].sum() == 0, "no ground truth"
    assert item["validity_mask"][0, :, :20].max() == 0 and item["validity_mask"][0, :, -20:].min() == 1, "no-data padding is detected"
    assert item["image_id"] == "x" and ds.has_ground_truth is False


def test_image_folder_without_images_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No images"):
        ImageFolderDataset(tmp_path, 5)


def test_identity_label_mapping_accepts_mvp_names_and_rejects_others():
    m = LabelMapping.identity_for(["Cracks", "Rust"], MVP)
    assert m.resolve("Rust") == "Rust"
    with pytest.raises(ValueError, match="Graffiti"):
        LabelMapping.identity_for(["Cracks", "Graffiti"], MVP)


def test_arbitrary_coco_dataset_with_identity_mapping(tmp_path):
    _write_jpg(tmp_path / "i.jpg")
    coco = {"images": [{"id": 1, "file_name": "i.jpg", "width": 80, "height": 60}],
            "categories": [{"id": 1, "name": "Cracks"}, {"id": 2, "name": "Rust"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "segmentation": [[10, 10, 50, 10, 50, 40, 10, 40]],
                             "area": 1200, "bbox": [10, 10, 40, 30], "iscrowd": 0}]}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    mapping = LabelMapping.identity_for([c["name"] for c in coco["categories"]], MVP)
    ds = CocoMultiLabelSegDataset(tmp_path, tmp_path / "coco.json", mapping, MVP, transform=build_eval_transform(512))
    item = ds[0]
    assert item["mask"][3].sum() > 0 and item["mask"][[0, 1, 2, 4]].sum() == 0, "Rust is channel 3 in MVP order"


def test_render_without_ground_truth_has_no_gt_column_or_scores():
    pred = np.zeros((3, H, W), bool)
    pred[1, 5:20, 5:20] = True
    fig = render_comparison(_image(), None, {"m1": pred, "m2": pred}, None, CLASSES, "new image", dpi=40)
    axes = [a for a in fig.axes if a.images]
    assert [a.get_title() for a in axes] == ["Image", "m1", "m2"], "Image + one column per model, no ground truth column"
    assert "Wetspot" in axes[1].get_xlabel() and "predicted" in axes[1].get_xlabel()
    assert "macro" not in axes[1].get_title(), "no scores can exist without labels"
    plt.close(fig)


def test_render_without_ground_truth_reports_empty_predictions():
    fig = render_comparison(_image(), None, {"m": np.zeros((3, H, W), bool)}, None, CLASSES, "t", dpi=40)
    assert "nothing" in [a for a in fig.axes if a.images][1].get_xlabel()
    plt.close(fig)


def test_legend_lists_only_the_classes():
    """No TP/FP/FN entries: the legend is the class colours (plus the no-data marker when padding exists)."""
    gt = _masks()
    valid = np.ones((H, W), bool)
    fig = render_comparison(_image(), gt, {"m": gt}, valid, CLASSES, "t", dpi=40)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == CLASSES
    valid[:, :5] = False
    fig2 = render_comparison(_image(), gt, {"m": gt}, valid, CLASSES, "t", dpi=40)
    assert [t.get_text() for t in fig2.legends[0].get_texts()] == CLASSES + ["no-data (excluded)"]
    plt.close(fig)
    plt.close(fig2)
