import pytest
import numpy as np
import torch

from metrics.boundary_metrics import boundary_f1, hard_cl_dice, skeleton_length_and_width
from metrics.instance_metrics import extract_instances, instance_full_mask, size_bucketed_detection_rate
from metrics.pixel_metrics import DEFAULT_THRESHOLD_GRID, PixelMetricAccumulator, ThresholdSweepAccumulator, sweep_per_class_thresholds
from metrics.presence_metrics import PresenceMetricAccumulator
from metrics.tiered_metrics import TieredAccumulator, _distance_near_mask, _iter_tiles

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


def _probs_target_for_fbeta_divergence():
    """10 GT-positive / 10 GT-negative pixels (4x5), hand-picked so the Dice(F1)-optimal threshold
    (0.7, fewer false positives) differs from the F2-optimal one (0.3, fewer missed defects):
    at t=0.3: TP=9 FP=8 FN=1 -> F1=16/24=0.667, F2=45/57=0.789
    at t=0.7: TP=6 FP=1 FN=4 -> F1=12/17=0.706, F2=30/47=0.638"""
    flat_target = [1] * 10 + [0] * 10
    flat_probs = [0.9] * 6 + [0.5] * 3 + [0.1] * 1 + [0.9] * 1 + [0.5] * 7 + [0.1] * 2
    target = torch.tensor(flat_target, dtype=torch.float32).view(1, 1, 4, 5)
    probs = torch.tensor(flat_probs, dtype=torch.float32).view(1, 1, 4, 5)
    return probs, target


def test_best_thresholds_defaults_to_dice_f1_unchanged():
    probs, target = _probs_target_for_fbeta_divergence()
    sweep = ThresholdSweepAccumulator(["a"], thresholds=np.array([0.3, 0.7]))
    sweep.update(probs, target)
    best = sweep.best_thresholds()["a"]
    assert best["threshold"] == pytest.approx(0.7)
    assert best["val_dice"] == pytest.approx(12 / 17, abs=1e-4)
    assert "val_f2" not in best, "beta=1.0 (default) must not add any new key to the output dict"


def test_best_thresholds_with_f2_beta_prefers_the_recall_weighted_threshold():
    probs, target = _probs_target_for_fbeta_divergence()
    sweep = ThresholdSweepAccumulator(["a"], thresholds=np.array([0.3, 0.7]))
    sweep.update(probs, target)
    best = sweep.best_thresholds(beta=2.0)["a"]
    assert best["threshold"] == pytest.approx(0.3), "F2 should pick the lower, more recall-favoring threshold"
    assert best["val_f2"] == pytest.approx(45 / 57, abs=1e-4)
    assert best["val_dice"] == pytest.approx(16 / 24, abs=1e-4), "val_dice always reports the actual Dice at the chosen threshold"


def test_fbeta_table_beta_1_matches_dice_table():
    probs, target = _probs_target_for_fbeta_divergence()
    sweep = ThresholdSweepAccumulator(["a"], thresholds=np.array([0.3, 0.7]))
    sweep.update(probs, target)
    assert np.allclose(sweep.fbeta_table(beta=1.0), sweep.dice_table())


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


def test_presence_accumulator_min_area_defaults_are_byte_identical_to_any_based_behavior():
    classes = ["a", "b"]
    rng = np.random.default_rng(0)
    pred = rng.random((2, 2, 6, 6)) > 0.8
    gt = rng.random((2, 2, 6, 6)) > 0.8
    default = PresenceMetricAccumulator(classes)
    explicit = PresenceMetricAccumulator(classes, min_gt_px=1, min_pred_px=1)
    for i in range(2):
        default.update(pred[i], gt[i])
        explicit.update(pred[i], gt[i])
    np.testing.assert_equal(default.compute(), explicit.compute())


def test_presence_min_area_thresholds_filter_small_noise():
    """A 3-pixel speck (annotation noise or a stray false positive) must not count as 'present'
    once a real minimum-area floor is set, while it still counts at the (default) 1px floor."""
    gt = np.zeros((1, 10, 10), bool)
    pred = np.zeros((1, 10, 10), bool)
    pred[0, 0, :3] = True  # 3px prediction speck, no GT at all
    lenient = PresenceMetricAccumulator(["a"])  # default min_pred_px=1
    lenient.update(pred, gt)
    assert lenient.compute()["per_class"]["a"]["fp"] == 1, "at the default floor, any nonzero area counts as present"

    strict = PresenceMetricAccumulator(["a"], min_gt_px=1, min_pred_px=10)
    strict.update(pred, gt)
    r = strict.compute()["per_class"]["a"]
    assert r["fp"] == 0 and r["tn"] == 1, "below min_pred_px, the speck is correctly ignored as absent"


def test_buffered_precision_recall_hand_computed():
    """GT = a 2x2 block at rows0-1/cols0-1; prediction = the same block shifted right by 1 column
    (2 of 4 pixels overlap). At buffer_px=0 the tolerance zone is exactly the mask itself, so
    precision/recall equal plain overlap (0.5 each). At buffer_px=1 (orthogonal-neighbor radius)
    every pixel of each mask falls within the other's tolerance zone, so both reach 1.0 — worked
    by hand in the plan, not just asserted to be 'better'."""
    gt = np.zeros((1, 4, 4), bool)
    gt[0, 0:2, 0:2] = True
    pred = np.zeros((1, 4, 4), bool)
    pred[0, 0:2, 1:3] = True

    acc0 = TieredAccumulator(["a"], [], buffer_px=0, min_gt_px=1, min_pred_px=1, tile_px=128)
    acc0.update(pred, gt)
    t2 = acc0.compute()["tier2"]["a"]
    assert t2["buffered_precision"] == pytest.approx(0.5) and t2["buffered_recall"] == pytest.approx(0.5)

    acc1 = TieredAccumulator(["a"], [], buffer_px=1, min_gt_px=1, min_pred_px=1, tile_px=128)
    acc1.update(pred, gt)
    t2 = acc1.compute()["tier2"]["a"]
    assert t2["buffered_precision"] == pytest.approx(1.0) and t2["buffered_recall"] == pytest.approx(1.0)


def test_tile_level_presence_catches_wrong_corner():
    """GT present only in the top-left 128px tile; prediction present only in the top-right one.
    Image-level presence calls this 'detected' (both sides have SOME pixel somewhere) even though
    the tile-level breakdown shows it's in the wrong place entirely: tp=0, fp=1, fn=1, tn=2 across
    the image's 4 tiles — exactly what tile-level adds over image-level."""
    gt = np.zeros((1, 256, 256), bool)
    gt[0, 0:10, 0:10] = True
    pred = np.zeros((1, 256, 256), bool)
    pred[0, 0:10, 138:148] = True  # top-right 128px tile

    acc = TieredAccumulator(["a"], [], buffer_px=1, min_gt_px=1, min_pred_px=1, tile_px=128)
    acc.update(pred, gt)
    result = acc.compute()
    img = result["tier1_image"]["per_class"]["a"]
    assert img["tp"] == 1, "image-level: both sides have damage somewhere -> counted as detected"
    tile = result["tier1_tile"]["per_class"]["a"]
    assert (tile["tp"], tile["fp"], tile["fn"], tile["tn"]) == (0, 1, 1, 2)
    assert result["tier1_tile"]["n_images"] == 4, "n_images counts TILES here, 4 per 256x256 image at tile_px=128"


def test_area_error_and_total_area_ratio_hand_computed():
    """3 images, GT area 10 each, prediction areas 12/8/20 -> per-image area errors 0.2/0.2/1.0
    (median 0.2, p90 by linear interpolation = 0.84) and a pooled total_area_ratio of 40/30."""
    acc = TieredAccumulator(["a"], [], buffer_px=0, min_gt_px=1, min_pred_px=1, tile_px=128)
    for gt_area, pred_area in [(10, 12), (10, 8), (10, 20)]:
        gt = np.zeros(100, bool)
        gt[:gt_area] = True
        pred = np.zeros(100, bool)
        pred[:pred_area] = True
        acc.update(pred.reshape(1, 10, 10), gt.reshape(1, 10, 10))
    t3 = acc.compute()["tier3"]["a"]
    assert t3["area_error_median"] == pytest.approx(0.2)
    assert t3["area_error_p90"] == pytest.approx(0.84)
    assert t3["total_area_ratio"] == pytest.approx(40 / 30)


def test_skeleton_length_and_width_of_a_known_straight_line():
    mask = np.zeros((20, 30), bool)
    mask[10, 0:20] = True  # a 1px-thick, 20px-long straight line: already its own skeleton
    length, width = skeleton_length_and_width(mask)
    assert length == pytest.approx(20.0)
    assert width == pytest.approx(1.0)


def test_skeleton_length_and_width_of_an_empty_mask_is_nan_width():
    length, width = skeleton_length_and_width(np.zeros((10, 10), bool))
    assert length == 0.0 and np.isnan(width)


def test_distance_near_mask_matches_hand_computed_buffer_shape():
    """Single foreground pixel at the center of a 5x5 grid: at buffer_px=1, only the 4 orthogonal
    neighbors (not the diagonals, Euclidean distance sqrt(2) > 1) plus the pixel itself are 'near'."""
    binary = np.zeros((5, 5), bool)
    binary[2, 2] = True
    near = _distance_near_mask(binary, buffer_px=1)
    expected = np.zeros((5, 5), bool)
    for r, c in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]:
        expected[r, c] = True
    assert np.array_equal(near, expected)
    assert not _distance_near_mask(np.zeros((5, 5), bool), buffer_px=5).any(), "nothing is 'near' an all-background mask"


def test_iter_tiles_covers_the_whole_image_including_uneven_edge_tiles():
    arr = np.zeros((2, 10, 7), bool)  # 7 is not a multiple of tile_px=4 -> an uneven edge column of tiles
    tiles = list(_iter_tiles(arr, tile_px=4))
    assert sum(t.shape[1] * t.shape[2] for t in tiles) == 10 * 7, "every pixel is covered exactly once, none padded/dropped"
    assert any(t.shape[2] == 3 for t in tiles), "the uneven edge tile is scored at its real (smaller) size, not padded"
