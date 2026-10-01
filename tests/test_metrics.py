import pytest
import numpy as np
import torch

from metrics.boundary_metrics import boundary_f1, hard_cl_dice
from metrics.instance_metrics import extract_instances, instance_full_mask, size_bucketed_detection_rate
from metrics.pixel_metrics import DEFAULT_THRESHOLD_GRID, PixelMetricAccumulator, ThresholdSweepAccumulator, sweep_per_class_thresholds

CLASSES = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]


def _target(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(4, 5, 32, 32, generator=g) > 0.85).float()


def test_gt_fed_back_as_prediction_is_perfect_for_every_class():
    t = _target()
    acc = PixelMetricAccumulator(CLASSES)
    acc.update(t, t)
    res = acc.compute()
    for name in CLASSES:
        assert abs(res["per_class"][name]["iou"] - 1.0) < 1e-4
        assert abs(res["per_class"][name]["dice"] - 1.0) < 1e-4
    assert abs(res["macro"]["iou"] - 1.0) < 1e-4 and abs(res["frequency_weighted"]["dice"] - 1.0) < 1e-4


def test_no_prediction_has_zero_recall():
    t = _target()
    acc = PixelMetricAccumulator(CLASSES)
    acc.update(torch.zeros_like(t), t)
    assert all(v["recall"] == 0 for v in acc.compute()["per_class"].values())


def test_half_cover_and_over_predict_known_values():
    tgt = torch.ones(1, 1, 1, 4)
    half = torch.zeros(1, 1, 1, 4)
    half[..., :2] = 1
    a = PixelMetricAccumulator(["x"])
    a.update(half, tgt)
    r = a.compute()["per_class"]["x"]
    assert abs(r["iou"] - 0.5) < 1e-4 and abs(r["dice"] - 2 / 3) < 1e-4 and abs(r["precision"] - 1) < 1e-4

    tgt2 = torch.zeros(1, 1, 1, 4)
    tgt2[..., 0] = 1
    b = PixelMetricAccumulator(["x"])
    b.update(torch.ones(1, 1, 1, 4), tgt2)
    r2 = b.compute()["per_class"]["x"]
    assert abs(r2["iou"] - 0.25) < 1e-4 and abs(r2["recall"] - 1) < 1e-4


def test_validity_mask_excludes_pixels_from_all_counts():
    t = _target()
    bad_pred = 1 - t
    validity = torch.zeros(4, 1, 32, 32)
    acc = PixelMetricAccumulator(CLASSES)
    acc.update(bad_pred, t, validity)
    assert acc.tp.sum() == 0 and acc.fp.sum() == 0 and acc.fn.sum() == 0 and acc.gt_pixels.sum() == 0


def test_accumulation_across_batches_equals_single_pass():
    t = _target()
    p = torch.rand_like(t)
    one = PixelMetricAccumulator(CLASSES)
    one.update(p, t)
    two = PixelMetricAccumulator(CLASSES)
    two.update(p[:2], t[:2])
    two.update(p[2:], t[2:])
    assert torch.allclose(one.tp, two.tp) and torch.allclose(one.fn, two.fn)


def test_frequency_weighted_uses_gt_pixel_counts():
    tgt = torch.zeros(1, 2, 4, 4)
    tgt[0, 0] = 1  # class a: 16 px, class b: 0 px
    pred = torch.zeros_like(tgt)
    pred[0, 0] = 1
    acc = PixelMetricAccumulator(["a", "b"])
    acc.update(pred, tgt)
    r = acc.compute()
    assert abs(r["frequency_weighted"]["iou"] - 1.0) < 1e-4
    assert r["macro"]["iou"] < 0.6  # class b (empty) drags the macro mean down


def test_threshold_sweep_finds_class_specific_optimum():
    n = 4
    tgt = torch.zeros(n, 2, 16, 16)
    tgt[:, :, :8] = 1
    probs = torch.zeros_like(tgt)
    probs[:, 0, :8] = 0.9
    probs[:, 0, 8:] = 0.2  # class 0 separable at any t in (0.2, 0.9)
    probs[:, 1, :8] = 0.4
    probs[:, 1, 8:] = 0.1  # class 1 needs a lower threshold
    th = sweep_per_class_thresholds(probs, tgt, ["a", "b"])
    assert th["b"] < 0.45 and 0.2 < th["a"] < 0.9


def test_default_threshold_grid_is_wider_than_the_old_005_to_095_grid():
    assert DEFAULT_THRESHOLD_GRID.min() < 0.05 and DEFAULT_THRESHOLD_GRID.max() > 0.95


def test_widened_grid_finds_a_true_optimum_outside_the_old_005_to_095_range():
    """Reproduces the real symptom seen on a finished run (effb0_task_pretrained_cif): a class
    whose true best threshold lies above 0.95, so the old grid could only ever report 0.95 (its
    edge) and never actually find it."""
    target = torch.zeros(1, 1, 10, 10)
    target[:, :, :5] = 1  # 50 true-positive pixels at prob 0.99
    probs = torch.zeros(1, 1, 10, 10)
    probs[:, :, :5] = 0.99
    probs[:, :, 5:8] = 0.97  # false-positive-prone pixels: only rejected above 0.97
    probs[:, :, 8:] = 0.10  # clean negatives

    old_grid = np.arange(0.05, 0.951, 0.05)  # the grid actually used before this fix
    old = ThresholdSweepAccumulator(["a"], thresholds=old_grid)
    old.update(probs, target)
    old_best = old.best_thresholds()["a"]
    assert old_best["threshold"] <= 0.95, "nothing in the old grid can exceed its own edge"

    new = ThresholdSweepAccumulator(["a"])  # DEFAULT_THRESHOLD_GRID
    new.update(probs, target)
    new_best = new.best_thresholds()["a"]
    assert new_best["threshold"] > 0.95, "the widened grid finds a threshold the old one could never reach"
    assert new_best["val_dice"] > old_best["val_dice"], "and it's a strictly better fit, not just a different one"


def test_hard_cldice_and_boundary_f1_identity_and_disjoint():
    m = np.zeros((50, 50), bool)
    m[10:40, 24:27] = True
    assert hard_cl_dice(m, m) > 0.99 and boundary_f1(m, m) > 0.99
    far = np.zeros_like(m)
    far[10:40, 5:8] = True
    assert hard_cl_dice(m, far) < 0.05 and boundary_f1(m, far) < 0.05
    assert hard_cl_dice(np.zeros_like(m), np.zeros_like(m)) == 1.0


def test_boundary_f1_tolerates_small_offset_but_not_large():
    m = np.zeros((60, 60), bool)
    m[20:40, 20:40] = True
    shifted1 = np.roll(m, 1, axis=1)
    shifted15 = np.roll(m, 15, axis=1)
    assert boundary_f1(m, shifted1, tolerance_px=2) > 0.9
    assert boundary_f1(m, shifted15, tolerance_px=2) < 0.6


def test_instance_detection_by_size_bucket():
    gt = np.zeros((300, 300), bool)
    gt[10:20, 10:20] = True  # small (100 px)
    gt[100:200, 100:200] = True  # large (10000 px)
    pred = np.zeros_like(gt)
    pred[100:200, 100:200] = True  # only the large one found
    r = size_bucketed_detection_rate(pred, gt)
    assert r.num_gt_instances["small"] == 1 and r.num_detected["small"] == 0
    assert r.num_gt_instances["large"] == 1 and r.num_detected["large"] == 1
    assert r.size_bucket_recall["large"] == 1.0 and np.isnan(r.size_bucket_recall["medium"])


def test_extract_instances_stores_crops_not_full_masks_and_respects_min_area():
    m = np.zeros((1000, 1000), bool)
    m[5:8, 5:8] = True  # area 9
    m[500:520, 500:520] = True  # area 400
    inst = extract_instances(m, min_area=10)
    assert len(inst) == 1 and inst[0]["area"] == 400
    assert inst[0]["crop"].shape == (20, 20), "must be bbox-cropped, never full-image (OOM on noisy predictions)"
    assert np.array_equal(instance_full_mask(inst[0], m.shape), np.pad(np.ones((20, 20), bool), ((500, 480), (500, 480))))


def test_many_components_do_not_blow_up_memory():
    rng = np.random.default_rng(0)
    noisy = rng.random((1300, 2200)) > 0.75  # sparse noise (below the 8-connected percolation threshold) -> ~10^5 tiny components
    inst = extract_instances(noisy, min_area=1)
    assert len(inst) > 50_000
    assert sum(i["crop"].nbytes for i in inst) < 200e6


def test_macro_present_ignores_classes_without_ground_truth():
    tgt = torch.zeros(1, 2, 4, 4)
    tgt[0, 0] = 1  # class a: perfect, class b: no GT at all
    acc = PixelMetricAccumulator(["a", "b"])
    acc.update(tgt.clone(), tgt)
    r = acc.compute()
    assert r["macro_present"]["dice"] == pytest.approx(1.0, abs=1e-4)
    assert r["macro"]["dice"] < 0.6, "macro still counts the empty class (unchanged behaviour)"
    assert np.isnan(PixelMetricAccumulator(["a"]).compute()["macro_present"]["dice"]), "no GT anywhere -> undefined, not 0"


def test_boundary_f1_of_full_image_masks_is_one_not_zero():
    """A mask covering the whole image has no inner boundary; identical full masks used to score 0 (found on
    dacl10k photos fully covered by 'Weathering')."""
    full = np.ones((32, 32), bool)
    assert boundary_f1(full, full) == pytest.approx(1.0)
    assert boundary_f1(full, np.zeros_like(full)) == 0.0, "a full mask vs. an empty prediction is still a miss"
    part = np.zeros_like(full)
    part[8:20, 8:20] = True
    assert boundary_f1(full, part) == 0.0, "full vs. partial: the full mask has no edge to match"
    assert boundary_f1(part, part) == pytest.approx(1.0, abs=1e-4)
