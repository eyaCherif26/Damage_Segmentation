import json

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.datasets.multilabel_seg_dataset import CocoMultiLabelSegDataset
from data.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from data.label_mapping import LabelMapping
from metrics.evaluation import (
    evaluate_external_with_tiled_inference,
    evaluate_split,
    resolve_class_subset,
    results_to_dataframe,
    sanity_check_gt_as_prediction,
    select_thresholds_on_val,
)
from metrics.pixel_metrics import ThresholdSweepAccumulator

CLASSES = ["Cracks", "Wetspot", "Spalling"]


class ToyDataset(Dataset):
    """Image channels ARE the ground truth, so an identity 'model' is a perfect predictor."""

    def __init__(self, n=6, source_split=3):
        g = torch.Generator().manual_seed(0)
        self.masks = []
        for i in range(n):
            m = torch.zeros(3, 64, 64)
            m[0, 5 + i : 9 + i, 10:50] = 1  # thin-ish horizontal crack
            m[1, 30:50, 30:50] = 1
            if i % 2:
                m[2, 0:20, 0:20] = 1
            self.masks.append(m)
        self.source_split = source_split

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, i):
        validity = torch.ones(1, 64, 64)
        validity[:, :, 60:] = 0  # 4 invalid columns
        return {
            "image": self.masks[i], "mask": self.masks[i], "validity_mask": validity, "image_id": i,
            "source_dataset": "a" if i < self.source_split else "b",
        }


class OracleModel(nn.Module):
    def forward(self, x):
        return x * 30 - 15  # sigmoid -> ~0/1 matching the mask


class InvertedModel(nn.Module):
    def forward(self, x):
        return -(x * 30 - 15)


def _loader():
    return DataLoader(ToyDataset(), batch_size=3)


def test_sweep_accumulator_picks_class_specific_thresholds_streaming():
    tgt = torch.zeros(2, 2, 8, 8)
    tgt[..., :4] = 1
    probs = torch.zeros_like(tgt)
    probs[:, 0, :, :4], probs[:, 0, :, 4:] = 0.9, 0.2
    probs[:, 1, :, :4], probs[:, 1, :, 4:] = 0.4, 0.1
    sweep = ThresholdSweepAccumulator(["a", "b"])
    sweep.update(probs[:1], tgt[:1])
    sweep.update(probs[1:], tgt[1:])  # streaming in two batches
    best = sweep.best_thresholds()
    assert best["b"]["threshold"] < 0.45 and 0.2 < best["a"]["threshold"] < 0.9
    assert best["a"]["val_dice"] == pytest.approx(1.0, abs=1e-4)


def test_class_without_gt_falls_back_to_default_threshold():
    sweep = ThresholdSweepAccumulator(["a", "b"])
    tgt = torch.zeros(1, 2, 4, 4)
    tgt[:, 0] = 1
    sweep.update(torch.rand(1, 2, 4, 4), tgt)
    best = sweep.best_thresholds(default=0.5)
    assert best["b"]["threshold"] == 0.5 and np.isnan(best["b"]["val_dice"])


def test_oracle_model_scores_perfectly_through_the_full_evaluation_path():
    dev = torch.device("cpu")
    th = select_thresholds_on_val(OracleModel(), _loader(), CLASSES, dev, amp=False)
    assert all(t["val_dice"] == pytest.approx(1.0, abs=1e-3) for t in th.values())
    flat = {c: t["threshold"] for c, t in th.items()}
    res = evaluate_split(OracleModel(), _loader(), CLASSES, ["Cracks"], flat, dev, amp=False)
    for c in CLASSES:
        assert res["overall"]["per_class"][c]["iou"] == pytest.approx(1.0, abs=1e-3)
    df = results_to_dataframe(res, CLASSES, th, "test")
    row = df[(df.scope == "overall") & (df["class"] == "Cracks")].iloc[0]
    assert row.boundary_f1 > 0.99 and row.cldice > 0.9
    assert row.inst_n_gt_small == len(ToyDataset()) and row.inst_recall_small == 1.0  # 4x40 px crack -> 'small' bucket
    assert np.isnan(row.inst_recall_large)
    assert {"source:a", "source:b"} <= set(df.scope)
    assert df[(df.scope == "overall") & (df["class"] == "Wetspot")].cldice.isna().all(), "clDice only for thin-structure classes"


def test_inverted_model_scores_near_zero():
    dev = torch.device("cpu")
    th = {c: {"threshold": 0.5, "val_dice": 0.0} for c in CLASSES}
    res = evaluate_split(InvertedModel(), _loader(), CLASSES, ["Cracks"], {c: 0.5 for c in CLASSES}, dev, amp=False)
    assert res["overall"]["macro"]["iou"] < 0.05


def test_invalid_columns_are_excluded_from_pixel_counts():
    dev = torch.device("cpu")
    res = evaluate_split(OracleModel(), _loader(), CLASSES, [], {c: 0.5 for c in CLASSES}, dev, amp=False)
    n = len(ToyDataset())
    assert res["overall"]["per_class"]["Wetspot"]["gt_pixels"] == n * 20 * 20  # wetspot block lies in valid columns
    assert res["overall"]["per_class"]["Cracks"]["gt_pixels"] == n * 4 * 40


def test_gt_as_prediction_sanity_check_passes_on_clean_data_and_detects_broken_metric(monkeypatch):
    dev = torch.device("cpu")
    sanity_check_gt_as_prediction(_loader(), CLASSES, dev)
    import metrics.evaluation as ev

    monkeypatch.setattr(ev, "boundary_f1", lambda p, g: 0.5)
    with pytest.raises(AssertionError):
        sanity_check_gt_as_prediction(_loader(), CLASSES, dev)


def test_resolve_class_subset_none_is_a_no_op():
    classes, indices = resolve_class_subset(CLASSES, None)
    assert classes == CLASSES and indices is None


def test_resolve_class_subset_keeps_checkpoint_order_regardless_of_requested_order():
    """--only_classes Spalling Cracks (reversed) must still come back in the checkpoint's own
    order — keeps colors/CSV columns/thresholds.json keys stable whether or not filtering is used."""
    classes, indices = resolve_class_subset(CLASSES, ["Spalling", "Cracks"])
    assert classes == ["Cracks", "Spalling"]
    assert indices == [0, 2]


def test_resolve_class_subset_rejects_unknown_name():
    with pytest.raises(ValueError, match="Fissure"):
        resolve_class_subset(CLASSES, ["Cracks", "Fissure"])


def test_instance_iou_threshold_changes_whether_a_partial_overlap_counts_as_detected():
    """A prediction overlapping a GT instance at IoU ~0.3 must count as 'not detected' at the
    default (0.5) threshold and 'detected' once the threshold is lowered below ~0.3 — this is the
    exact mechanism behind a real observation: pixel Dice/recall can be modestly positive (a
    reasonable-looking overlay) while inst_recall_* reads exactly 0, because the default IoU bar
    for 'detected' is much stricter than a human's visual judgment of 'found it'."""
    from metrics.evaluation import _PerImageMetrics

    gt = np.zeros((1, 20, 20), dtype=bool)
    gt[0, 0:10, 0:10] = True  # 100 px GT instance
    pred = np.zeros((1, 20, 20), dtype=bool)
    pred[0, 0:10, 0:3] = True  # entirely inside gt: intersection=30, pred area=30, union=100+30-30=100 -> IoU=0.30

    strict = _PerImageMetrics(["Cracks"], thin_classes=[])
    strict.update(pred, gt)
    assert strict.inst_hits["Cracks"]["small"] == 0, "IoU 0.30 must NOT count as detected at the default 0.5 threshold"

    lenient = _PerImageMetrics(["Cracks"], thin_classes=[], instance_iou_threshold=0.2)
    lenient.update(pred, gt)
    assert lenient.inst_hits["Cracks"]["small"] == 1, "the SAME prediction must count as detected once the bar is lowered below its IoU"


def test_predict_channel_indices_slices_both_probs_and_target_consistently():
    batch = {"image": torch.arange(2 * 3 * 4 * 4.0).reshape(2, 3, 4, 4), "mask": torch.arange(2 * 3 * 4 * 4.0).reshape(2, 3, 4, 4) + 100,
             "validity_mask": torch.ones(2, 1, 4, 4)}

    class Identity(nn.Module):
        def forward(self, x):
            return x

    from metrics.evaluation import _predict
    probs, target, _ = _predict(Identity(), batch, torch.device("cpu"), False, channel_indices=[0, 2])
    assert probs.shape == (2, 2, 4, 4) and target.shape == (2, 2, 4, 4)
    full_probs, full_target, _ = _predict(Identity(), batch, torch.device("cpu"), False)
    assert torch.equal(probs, full_probs[:, [0, 2]]) and torch.equal(target, full_target[:, [0, 2]])


def test_evaluate_split_with_channel_indices_scores_only_the_selected_classes():
    """Filtering to (Cracks, Spalling) via channel_indices=[0,2] must give exactly the pixel
    counts/scores those two channels actually have in ToyDataset (known from its construction,
    same source used by test_invalid_columns_are_excluded_from_pixel_counts for Cracks/Wetspot) —
    proves the slice inside _predict is really selecting those channels, not just relabeling."""
    dev = torch.device("cpu")
    filtered_classes, indices = resolve_class_subset(CLASSES, ["Cracks", "Spalling"])
    assert filtered_classes == ["Cracks", "Spalling"] and indices == [0, 2]
    res = evaluate_split(OracleModel(), _loader(), filtered_classes, [], {c: 0.5 for c in filtered_classes},
                         dev, amp=False, channel_indices=indices)
    n = len(ToyDataset())
    assert res["overall"]["per_class"]["Cracks"]["gt_pixels"] == n * 4 * 40
    assert res["overall"]["per_class"]["Spalling"]["gt_pixels"] == (n // 2) * 20 * 20  # only odd i has this block
    for c in filtered_classes:
        assert res["overall"]["per_class"][c]["dice"] == pytest.approx(1.0, abs=1e-3), "OracleModel is a perfect predictor"


# ---------------------------------------------------------------------- evaluate_external_with_tiled_inference

class _TileOracleModel(nn.Module):
    """Inverts run_tiled_inference's own ImageNet normalization to recover the ~[0,1] pixel value,
    then maps it to a confident logit — same 'the image IS the mask' trick as OracleModel above,
    adapted because this path never goes through a dataset __getitem__ tensor: run_tiled_inference
    does its own normalization internally on raw tiles cut from the full-resolution image."""

    def forward(self, x):
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * std + mean) * 30 - 15


def _make_large_encoded_dataset(tmp_path, size=(900, 700)):
    """A synthetic image whose RGB channels literally ARE 2 classes' ground truth (255 where that
    class is present; the 3rd class, Spalling, is never drawn), with a matching COCO file — so an
    oracle model that just reads the image back is a perfect predictor. The two blocks are placed
    far enough apart (and the image large enough) that no single training-size tile can see both:
    correctness here depends on real tiling + stitching, not one lucky whole-image forward pass."""
    w, h = size
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[50:300, 50:300, 0] = 255     # Cracks block, top-left (RGB order — see _load_image's BGR2RGB)
    mask[400:650, 400:650, 1] = 255   # Wetspot block, bottom-right — a different tile entirely
    cv2.imwrite(str(img_dir / "big.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))  # imwrite expects BGR

    def poly(x0, y0, x1, y1):
        return [x0, y0, x1, y0, x1, y1, x0, y1]

    coco = {
        "images": [{"id": 1, "file_name": "big.png", "width": w, "height": h}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [poly(50, 50, 300, 300)],
             "bbox": [50, 50, 250, 250], "area": 62500, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [poly(400, 400, 650, 650)],
             "bbox": [400, 400, 250, 250], "area": 62500, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "Cracks"}, {"id": 2, "name": "Wetspot"}, {"id": 3, "name": "Spalling"}],
    }
    ann = tmp_path / "annotations.json"
    ann.write_text(json.dumps(coco))
    return ann, img_dir


def test_evaluate_external_with_tiled_inference_scores_perfectly_for_an_oracle_model(tmp_path):
    """Correctness of the whole path added for the 'evaluate an external dataset the way real
    deployment inference works' request: full-resolution GT rasterization, run_tiled_inference's
    tiling/overlap-blending, threshold application, and PixelMetricAccumulator/_PerImageMetrics
    accumulation, end to end — not just 'the CLI doesn't crash' (see test_evaluate_external.py)."""
    classes = ["Cracks", "Wetspot", "Spalling"]
    ann, img_dir = _make_large_encoded_dataset(tmp_path)
    mapping = LabelMapping.identity_for([c["name"] for c in json.loads(ann.read_text())["categories"]], classes)
    dataset = CocoMultiLabelSegDataset(img_dir, ann, mapping, classes, transform=None, source_name="big", detect_no_data=False)

    dev = torch.device("cpu")
    result = evaluate_external_with_tiled_inference(
        _TileOracleModel(), dataset, classes, ["Cracks"], {c: 0.5 for c in classes}, dev,
        tile_size=256, overlap=32, batch_size=4,
    )
    for c in ("Cracks", "Wetspot"):
        assert result["overall"]["per_class"][c]["iou"] == pytest.approx(1.0, abs=1e-2), c
    assert result["overall"]["per_class"]["Cracks"]["gt_pixels"] == 250 * 250
    # Spalling never appears in GT and the oracle correctly predicts it absent everywhere — iou is
    # exactly 0 for tp=fp=fn=0 (no epsilon in the numerator, see metrics/pixel_metrics.py), by this
    # project's own established convention (macro_present exists specifically to exclude this).
    assert result["overall"]["per_class"]["Spalling"]["gt_pixels"] == 0
    assert result["overall"]["per_class"]["Spalling"]["iou"] == 0.0


def test_evaluate_external_with_tiled_inference_respects_channel_indices(tmp_path):
    ann, img_dir = _make_large_encoded_dataset(tmp_path)
    all_classes = ["Cracks", "Wetspot", "Spalling"]
    mapping = LabelMapping.identity_for([c["name"] for c in json.loads(ann.read_text())["categories"]], all_classes)
    dataset = CocoMultiLabelSegDataset(img_dir, ann, mapping, all_classes, transform=None, source_name="big", detect_no_data=False)

    dev = torch.device("cpu")
    filtered, indices = resolve_class_subset(all_classes, ["Cracks", "Spalling"])
    result = evaluate_external_with_tiled_inference(
        _TileOracleModel(), dataset, filtered, [], {c: 0.5 for c in filtered}, dev,
        tile_size=256, overlap=32, batch_size=4, channel_indices=indices,
    )
    assert set(result["overall"]["per_class"]) == {"Cracks", "Spalling"}, "Wetspot must not appear at all when filtered out"
    assert result["overall"]["per_class"]["Cracks"]["iou"] == pytest.approx(1.0, abs=1e-2)
