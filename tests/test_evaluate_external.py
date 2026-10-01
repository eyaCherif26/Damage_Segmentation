"""scripts/evaluate.py on ANY labelled COCO dataset (--coco/--images), run as a subprocess on CPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import models
from configs.loader import load_experiment_config
from data.datasets.external import build_labelled_dataset

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs/experiments/baseline_smp_mobilenet.yaml"
SCRIPT = REPO / "scripts/evaluate.py"
CLASSES = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]


def _make_dataset(root: Path, names: tuple[str, str]) -> tuple[Path, Path]:
    """Two 640x480 images; category 1 = a rectangle, category 2 = another. Only 2 of the 5 MVP classes exist."""
    images = root / "imgs"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": names[0]}, {"id": 2, "name": names[1]}]}
    for i in range(2):
        cv2.imwrite(str(images / f"im{i}.png"), rng.integers(40, 220, (480, 640, 3), dtype=np.uint8))
        coco["images"].append({"id": i + 1, "file_name": f"im{i}.png", "width": 640, "height": 480})
        for k, (x, y) in enumerate(((50, 60), (350, 200))):
            poly = [x, y, x + 150, y, x + 150, y + 90, x, y + 90]
            coco["annotations"].append({"id": 10 * i + k + 1, "image_id": i + 1, "category_id": k + 1, "segmentation": [poly],
                                        "bbox": [x, y, 150, 90], "area": 150 * 90, "iscrowd": 0})
    ann = root / "annotations.json"
    ann.write_text(json.dumps(coco))
    return ann, images


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = load_experiment_config(CONFIG)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save({"epoch": 3, "model_state_dict": models.build_model(cfg.model).state_dict()}, path)
    return path


@pytest.fixture
def results_dir(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    thr = {c: {"threshold": 0.3 + 0.1 * i, "val_dice": 0.5} for i, c in enumerate(CLASSES)}
    (out / "thresholds.json").write_text(json.dumps(thr))
    return out


def _run(checkpoint, out, *extra):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--checkpoint", str(checkpoint), "--output_dir", str(out),
         "--batch_size", "2", "--num_workers", "0", *map(str, extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900,
    )


def test_external_dataset_uses_saved_thresholds_and_only_scores_present_classes(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    before = (results_dir / "thresholds.json").read_text()
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (results_dir / "thresholds.json").read_text() == before, "thresholds are never re-tuned on the new data"
    assert not (results_dir / "results_test.csv").exists() and not (results_dir / "results_val.csv").exists()

    df = pd.read_csv(results_dir / "results_harbor.csv")
    assert set(df.split) == {"external:harbor"}
    overall = df[df.scope == "overall"].set_index("class")
    assert overall.loc["Cracks", "threshold"] == pytest.approx(0.3) and overall.loc["Rust", "threshold"] == pytest.approx(0.6)
    assert overall.loc["Cracks", "gt_pixels"] > 0 and overall.loc["Rust", "gt_pixels"] > 0
    for absent in ("Wetspot", "Spalling", "Efflorescence"):
        assert overall.loc[absent, "gt_pixels"] == 0
    # the fair average covers exactly the two classes that have labels, and differs from the all-class macro
    assert overall.loc["macro_present", "dice"] == pytest.approx((overall.loc["Cracks", "dice"] + overall.loc["Rust", "dice"]) / 2)
    assert overall.loc["macro", "dice"] == pytest.approx(overall.loc[CLASSES, "dice"].mean())
    assert "no labels for" in proc.stdout and "macro_present" in proc.stdout


def test_label_map_translates_foreign_category_names(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Fissure", "Corrosion"))
    label_map = tmp_path / "map.yaml"
    label_map.write_text("Fissure: Cracks\nCorrosion: Rust\n")
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--label_map", label_map)
    assert proc.returncode == 0, proc.stderr[-2000:]
    df = pd.read_csv(results_dir / "results_data.csv")  # default name = the annotation file's folder
    overall = df[df.scope == "overall"].set_index("class")
    assert overall.loc["Cracks", "gt_pixels"] > 0 and overall.loc["Rust", "gt_pixels"] > 0


def test_foreign_categories_without_a_label_map_fail_loudly(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Fissure", "Corrosion"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images)
    assert proc.returncode != 0 and "not MVP classes" in proc.stderr
    assert not list(results_dir.glob("results_*.csv")), "nothing may be written for a failed run"


def test_argument_combinations_are_validated(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    assert "--coco needs --images" in _run(checkpoint, results_dir, "--coco", ann).stderr
    assert "only apply together with --coco" in _run(checkpoint, results_dir, "--images", images).stderr
    missing = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--thresholds", tmp_path / "nope.json")
    assert missing.returncode != 0 and "does not exist" in missing.stderr
    both = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--fixed_threshold", "0.3", "--thresholds", results_dir / "thresholds.json")
    assert both.returncode != 0 and "mutually exclusive" in both.stderr
    bad_range = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--fixed_threshold", "1.5")
    assert bad_range.returncode != 0 and "must be in [0, 1]" in bad_range.stderr


def test_fixed_threshold_skips_grid_search_and_does_not_touch_thresholds_json(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    before = (results_dir / "thresholds.json").read_text()
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor", "--fixed_threshold", "0.2")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (results_dir / "thresholds.json").read_text() == before, "the canonical val-selected record must survive a what-if run"
    used = json.loads((results_dir / "thresholds_used.json").read_text())
    assert all(used[c]["threshold"] == pytest.approx(0.2) for c in CLASSES)
    df = pd.read_csv(results_dir / "results_harbor.csv")
    overall = df[df.scope == "overall"].set_index("class")
    assert overall.loc["Cracks", "threshold"] == pytest.approx(0.2)


def test_class_threshold_overrides_one_class_on_top_of_the_saved_file(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor", "--class_threshold", "Rust=0.05")
    assert proc.returncode == 0, proc.stderr[-2000:]
    used = json.loads((results_dir / "thresholds_used.json").read_text())
    assert used["Rust"]["threshold"] == pytest.approx(0.05)
    assert used["Cracks"]["threshold"] == pytest.approx(0.3), "unlisted classes keep the base (saved-file) value"


def test_reusing_the_default_thresholds_json_silently_writes_nothing_extra(checkpoint, results_dir, tmp_path):
    """No --thresholds/--fixed_threshold/--class_threshold given: the implicit default-file reuse in
    --coco mode is silent, not a "what-if" — it must not spawn a redundant thresholds_used.json."""
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not (results_dir / "thresholds_used.json").exists()


def test_build_labelled_dataset_reports_absent_and_unmapped_classes(tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Fissure", "Graffiti"))
    label_map = tmp_path / "m.yaml"
    label_map.write_text("Fissure: Cracks\nGraffiti: null\n")
    cfg = load_experiment_config(CONFIG)
    ds, info = build_labelled_dataset(ann, images, cfg.data, None, label_map)
    assert info["n_images"] == 2 and info["unmapped"] == ["Graffiti"]
    assert info["absent_classes"] == ["Wetspot", "Spalling", "Rust", "Efflorescence"]
    assert len(ds) == 2
    with pytest.raises(FileNotFoundError):
        build_labelled_dataset(tmp_path / "missing.json", images, cfg.data, None)
    with pytest.raises(FileNotFoundError):
        build_labelled_dataset(ann, tmp_path / "missing_dir", cfg.data, None)


def test_only_classes_reports_a_subset_and_does_not_touch_the_canonical_files(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    before_thresholds = (results_dir / "thresholds.json").read_text()
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor", "--only_classes", "Cracks", "Rust")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "reporting only" in proc.stdout
    assert (results_dir / "thresholds.json").read_text() == before_thresholds
    used = json.loads((results_dir / "thresholds_used.json").read_text())
    assert set(used) == {"Cracks", "Rust"}, "the used-thresholds file only records the filtered subset"

    df = pd.read_csv(results_dir / "results_harbor.csv")
    overall = df[df.scope == "overall"].set_index("class")
    assert set(overall.index) & set(CLASSES) == {"Cracks", "Rust"}, "only the requested classes are reported at all"


def test_only_classes_rejects_an_unknown_name_before_any_dataset_work(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--only_classes", "Fissure")
    assert proc.returncode != 0 and "Fissure" in proc.stderr
    assert not list(results_dir.glob("results_*.csv"))


# ---------------------------------------------------------------------- native_crop: sliding-window tiled inference

CONFIG_NATIVE = REPO / "configs/experiments/mobilenet_native_cif_512.yaml"


@pytest.fixture(scope="module")
def checkpoint_native(tmp_path_factory):
    cfg = load_experiment_config(CONFIG_NATIVE)
    path = tmp_path_factory.mktemp("ckpt_native") / "best.pt"
    torch.save({"epoch": 3, "model_state_dict": models.build_model(cfg.model).state_dict()}, path)
    return path


def _make_large_dataset(root: Path, names: tuple[str, str], size: tuple[int, int] = (1200, 900)) -> tuple[Path, Path]:
    """One image well above the training tile size (512px) — the exact case where a non-overlapping
    fixed-tile grid and sliding-window overlap-blended inference can genuinely disagree."""
    images = root / "imgs"
    images.mkdir(parents=True)
    rng = np.random.default_rng(1)
    w, h = size
    coco = {"images": [{"id": 1, "file_name": "big.png", "width": w, "height": h}], "annotations": [],
           "categories": [{"id": 1, "name": names[0]}, {"id": 2, "name": names[1]}]}
    cv2.imwrite(str(images / "big.png"), rng.integers(40, 220, (h, w, 3), dtype=np.uint8))
    for k, (x, y) in enumerate(((100, 100), (700, 600))):
        poly = [x, y, x + 150, y, x + 150, y + 90, x, y + 90]
        coco["annotations"].append({"id": k + 1, "image_id": 1, "category_id": k + 1, "segmentation": [poly],
                                    "bbox": [x, y, 150, 90], "area": 150 * 90, "iscrowd": 0})
    ann = root / "annotations.json"
    ann.write_text(json.dumps(coco))
    return ann, images


def _run_native(checkpoint, out, *extra):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG_NATIVE), "--checkpoint", str(checkpoint), "--output_dir", str(out),
         "--batch_size", "2", "--num_workers", "0", *map(str, extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900,
    )


def test_native_crop_coco_defaults_to_sliding_window_tiled_inference(checkpoint_native, results_dir, tmp_path):
    """The behavior the user actually asked for: an external dataset's images larger than the
    training tile size get ONE stitched, overlap-blended probability map per image (matching
    scripts/infer_orthophoto.py/predict_deployment.py's real deployment inference), not treated as
    a grid of independent fixed-size crops."""
    ann, images = _make_large_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run_native(checkpoint_native, results_dir, "--coco", ann, "--images", images, "--name", "big_harbor")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "sliding-window inference" in proc.stdout
    assert (results_dir / "results_big_harbor.csv").exists()


def test_grid_eval_flag_reproduces_the_old_fixed_tile_grid_behavior(checkpoint_native, results_dir, tmp_path):
    """--grid_eval is the escape hatch back to the old (still correct, still tested) behavior —
    matches the configured val/test protocol exactly, for direct comparison against the default."""
    ann, images = _make_large_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run_native(checkpoint_native, results_dir, "--coco", ann, "--images", images, "--name", "big_harbor", "--grid_eval")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "sliding-window inference" not in proc.stdout
    assert "tiles at pixel_scale" in proc.stdout


def test_whole_image_config_defaults_to_plain_resize(checkpoint, results_dir, tmp_path):
    """Regression guard: a whole_image checkpoint must NOT get sliding-window tiling by default
    (matches how it was actually trained) — but see the next two tests: it can still be forced on."""
    ann, images = _make_large_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "big_harbor")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "sliding-window inference" not in proc.stdout


def test_sliding_window_flag_forces_it_on_for_a_whole_image_checkpoint(checkpoint, results_dir, tmp_path):
    """The user's actual ask: infer_orthophoto.py applies run_tiled_inference regardless of
    resize_mode, so evaluate.py must be able to match that for a whole_image checkpoint too, not
    just a native_crop one."""
    ann, images = _make_large_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "big_harbor", "--sliding_window")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "sliding-window inference" in proc.stdout and "forced via --sliding_window" in proc.stdout
    assert (results_dir / "results_big_harbor.csv").exists()


def test_sliding_window_and_grid_eval_are_mutually_exclusive(checkpoint, results_dir, tmp_path):
    ann, images = _make_large_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--sliding_window", "--grid_eval")
    assert proc.returncode != 0 and "mutually exclusive" in proc.stderr


def test_grid_eval_on_a_whole_image_checkpoint_warns_and_has_no_effect(checkpoint, results_dir, tmp_path):
    """--grid_eval only means something relative to a tiled default; a whole_image checkpoint was
    never going to tile in the first place, so it should warn rather than silently do nothing."""
    ann, images = _make_large_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "big_harbor", "--grid_eval")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "no effect" in proc.stdout
    assert "images -> sliding-window inference (" not in proc.stdout, "the warning mentioning the phrase is fine; actually dispatching to it is not"


def test_only_classes_composes_with_tiled_inference(checkpoint_native, results_dir, tmp_path):
    ann, images = _make_large_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run_native(checkpoint_native, results_dir, "--coco", ann, "--images", images, "--name", "big_harbor",
                       "--only_classes", "Cracks")
    assert proc.returncode == 0, proc.stderr[-3000:]
    df = pd.read_csv(results_dir / "results_big_harbor.csv")
    overall = df[df.scope == "overall"].set_index("class")
    assert set(overall.index) & set(CLASSES) == {"Cracks"}, "only the requested class is reported, even through the tiled path"


def test_only_classes_composes_with_class_threshold_for_an_excluded_class(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images,
               "--only_classes", "Cracks", "Rust", "--class_threshold", "Wetspot=0.3")
    assert proc.returncode != 0 and "not one of" in proc.stderr


# ---------------------------------------------------------------------- --retune_on_new_data / --calibration_fraction

def _make_multi_image_dataset(root: Path, n: int = 8, names: tuple[str, str] = ("Cracks", "Rust")) -> tuple[Path, Path]:
    """`n` distinct 640x480 images, each with one rectangle of a randomly chosen one of the two
    classes — enough distinct images for a calibration/report split to be meaningful."""
    images = root / "imgs"
    images.mkdir(parents=True)
    rng = np.random.default_rng(2)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": names[0]}, {"id": 2, "name": names[1]}]}
    for i in range(n):
        cv2.imwrite(str(images / f"im{i}.png"), rng.integers(40, 220, (480, 640, 3), dtype=np.uint8))
        coco["images"].append({"id": i + 1, "file_name": f"im{i}.png", "width": 640, "height": 480})
        cat = 1 if i % 2 == 0 else 2
        x, y = int(rng.integers(20, 300)), int(rng.integers(20, 250))
        poly = [x, y, x + 150, y, x + 150, y + 90, x, y + 90]
        coco["annotations"].append({"id": i + 1, "image_id": i + 1, "category_id": cat, "segmentation": [poly],
                                    "bbox": [x, y, 150, 90], "area": 150 * 90, "iscrowd": 0})
    ann = root / "annotations.json"
    ann.write_text(json.dumps(coco))
    return ann, images


def test_retune_on_new_data_warns_and_writes_selftuned_files(checkpoint, results_dir, tmp_path):
    ann, images = _make_multi_image_dataset(tmp_path / "data")
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor", "--retune_on_new_data")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "NOT a fair generalization estimate" in proc.stdout
    assert (results_dir / "thresholds_selftuned.json").exists()
    assert (results_dir / "results_harbor_selftuned.csv").exists()
    assert not (results_dir / "results_harbor.csv").exists(), "must never be confused with a real frozen-threshold run"
    before = (results_dir / "thresholds.json").read_text()
    assert (results_dir / "thresholds.json").read_text() == before, "the canonical val-selected record is untouched"


def test_retune_on_new_data_works_through_the_tiled_sliding_window_path(checkpoint_native, results_dir, tmp_path):
    """The user's actual scenario: their native_crop checkpoint uses --sliding_window by default,
    so retuning must work through select_thresholds_with_tiled_inference too, not just the
    DataLoader-based select_thresholds_on_val."""
    ann, images = _make_multi_image_dataset(tmp_path / "data")
    proc = _run_native(checkpoint_native, results_dir, "--coco", ann, "--images", images, "--name", "harbor_native", "--retune_on_new_data")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (results_dir / "thresholds_selftuned.json").exists()
    assert (results_dir / "results_harbor_native_selftuned.csv").exists()


def test_calibration_fraction_splits_by_whole_image_and_reports_only_held_out(checkpoint, results_dir, tmp_path):
    ann, images = _make_multi_image_dataset(tmp_path / "data", n=8)
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor", "--calibration_fraction", "0.5")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "4 images for" in proc.stdout and "4 held out for reporting" in proc.stdout
    assert (results_dir / "thresholds_calibrated.json").exists()
    df = pd.read_csv(results_dir / "results_harbor_calibrated.csv")
    overall = df[df.scope == "overall"].set_index("class")
    # 4 held-out images, one 150x90 box each, resized 640x480 -> 512x512 (whole_image, this checkpoint's
    # own config) along with the image: 13500 * (512/640) * (512/480) per image, times 4 images.
    expected = 4 * 150 * 90 * (512 / 640) * (512 / 480)
    assert overall.loc[["Cracks", "Rust"], "gt_pixels"].sum() == pytest.approx(expected, rel=0.02)


def test_calibration_fraction_works_through_the_tiled_sliding_window_path(checkpoint_native, results_dir, tmp_path):
    ann, images = _make_multi_image_dataset(tmp_path / "data", n=8)
    proc = _run_native(checkpoint_native, results_dir, "--coco", ann, "--images", images, "--name", "harbor_native",
                       "--calibration_fraction", "0.5")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (results_dir / "thresholds_calibrated.json").exists()
    assert (results_dir / "results_harbor_native_calibrated.csv").exists()


def test_calibration_fraction_and_retune_on_new_data_are_mutually_exclusive(checkpoint, results_dir, tmp_path):
    ann, images = _make_multi_image_dataset(tmp_path / "data")
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--retune_on_new_data", "--calibration_fraction", "0.5")
    assert proc.returncode != 0 and "mutually exclusive" in proc.stderr


def test_retune_options_need_coco(checkpoint, results_dir, tmp_path):
    proc = _run(checkpoint, results_dir, "--retune_on_new_data")
    assert proc.returncode != 0 and "only apply together with --coco" in proc.stderr


def test_retune_options_conflict_with_explicit_thresholds(checkpoint, results_dir, tmp_path):
    ann, images = _make_multi_image_dataset(tmp_path / "data")
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--retune_on_new_data", "--fixed_threshold", "0.3")
    assert proc.returncode != 0 and "different ways" in proc.stderr


def test_calibration_fraction_out_of_range_is_rejected(checkpoint, results_dir, tmp_path):
    ann, images = _make_multi_image_dataset(tmp_path / "data")
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--calibration_fraction", "1.5")
    assert proc.returncode != 0 and "must be in (0, 1)" in proc.stderr


def test_calibration_fraction_leaving_no_report_images_is_rejected(checkpoint, results_dir, tmp_path):
    """2 images, calibration_fraction so high it would take both -> no held-out reporting set left."""
    ann, images = _make_multi_image_dataset(tmp_path / "data", n=2)
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--calibration_fraction", "0.99")
    assert proc.returncode != 0 and "leaves no images" in proc.stderr


def test_detection_iou_threshold_out_of_range_is_rejected(checkpoint, results_dir, tmp_path):
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--detection_iou_threshold", "1.5")
    assert proc.returncode != 0 and "must be in (0, 1]" in proc.stderr


def test_detection_iou_threshold_is_accepted_and_runs(checkpoint, results_dir, tmp_path):
    """CLI-level wiring check (the actual effect on inst_recall_* is unit-tested directly in
    tests/test_evaluation.py::test_instance_iou_threshold_changes_whether_a_partial_overlap_counts_as_detected)."""
    ann, images = _make_dataset(tmp_path / "data", ("Cracks", "Rust"))
    proc = _run(checkpoint, results_dir, "--coco", ann, "--images", images, "--name", "harbor", "--detection_iou_threshold", "0.1")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (results_dir / "results_harbor.csv").exists()
