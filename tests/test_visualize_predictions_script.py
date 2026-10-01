"""scripts/visualize_predictions.py threshold options (--fixed_threshold, --class_threshold,
--compare_thresholds, --only_classes), run as a subprocess on CPU against a small synthetic COCO
dataset."""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

import models
from configs.loader import load_experiment_config

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs/experiments/baseline_smp_mobilenet.yaml"
SCRIPT = REPO / "scripts/visualize_predictions.py"
CLASSES = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]


def _make_dataset(root: Path) -> tuple[Path, Path]:
    """Two 640x480 images, one "Cracks" rectangle each — just enough for a real forward pass + GT overlay."""
    images = root / "imgs"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "Cracks"}]}
    for i in range(2):
        cv2.imwrite(str(images / f"im{i}.png"), rng.integers(40, 220, (480, 640, 3), dtype=np.uint8))
        coco["images"].append({"id": i + 1, "file_name": f"im{i}.png", "width": 640, "height": 480})
        poly = [50, 60, 200, 60, 200, 150, 50, 150]
        coco["annotations"].append({"id": i + 1, "image_id": i + 1, "category_id": 1, "segmentation": [poly],
                                    "bbox": [50, 60, 150, 90], "area": 150 * 90, "iscrowd": 0})
    ann = root / "annotations.json"
    ann.write_text(json.dumps(coco))
    return ann, images


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = load_experiment_config(CONFIG)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save({"epoch": 3, "model_state_dict": models.build_model(cfg.model).state_dict()}, path)
    return path


def _run(checkpoint, out, model_spec, *extra):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG), str(checkpoint), *model_spec,
         "--output_dir", str(out), "--num_images", "2", "--seed", "0", *map(str, extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )


def test_bare_number_as_model_threshold_arg_applies_to_every_class(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    proc = _run(checkpoint, tmp_path / "out", ["0.3"], "--coco", ann, "--images", images)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "fixed threshold 0.3 for every class" in proc.stdout
    assert (tmp_path / "out" / "overview.png").exists()


def test_fixed_threshold_flag_overrides_every_model(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    proc = _run(checkpoint, tmp_path / "out", [], "--coco", ann, "--images", images, "--fixed_threshold", "0.2")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--fixed_threshold 0.2 applied to every class of every model" in proc.stdout


def test_class_threshold_overrides_a_single_class(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    proc = _run(checkpoint, tmp_path / "out", [], "--coco", ann, "--images", images, "--class_threshold", "Rust=0.05")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "overrides applied on top: {'Rust': 0.05}" in proc.stdout


def test_compare_thresholds_renders_one_extra_column_per_value(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    out = tmp_path / "out"
    proc = _run(checkpoint, out, [], "--coco", ann, "--images", images, "--compare_thresholds", "0.1", "0.5", "0.9")
    assert proc.returncode == 0, proc.stderr[-2000:]
    selection = json.loads((out / "selection.json").read_text())
    # 3 base image columns (Image/GT/base) would be 3 cols normally; --compare_thresholds adds 3 more (0.1/0.5/0.9)
    assert "th=0.1" in proc.stdout and "th=0.5" in proc.stdout and "th=0.9" in proc.stdout
    assert len(selection["images"]) == 2


def test_compare_thresholds_rejects_more_than_one_model(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG), str(checkpoint), "--model", str(CONFIG), str(checkpoint),
         "--coco", str(ann), "--images", str(images), "--compare_thresholds", "0.1", "0.5",
         "--output_dir", str(tmp_path / "out"), "--num_images", "2"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0 and "needs exactly one --model" in proc.stderr


def test_fixed_threshold_out_of_range_is_rejected(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    proc = _run(checkpoint, tmp_path / "out", [], "--coco", ann, "--images", images, "--fixed_threshold", "1.5")
    assert proc.returncode != 0 and "must be in [0, 1]" in proc.stderr


def test_only_classes_renders_a_subset_for_a_single_model(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    out = tmp_path / "out"
    proc = _run(checkpoint, out, [], "--coco", ann, "--images", images, "--only_classes", "Cracks", "Rust")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "rendering only ['Cracks', 'Rust']" in proc.stdout


def test_only_classes_rejects_an_unknown_name(checkpoint, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    proc = _run(checkpoint, tmp_path / "out", [], "--coco", ann, "--images", images, "--only_classes", "Fissure")
    assert proc.returncode != 0 and "Fissure" in proc.stderr


@pytest.fixture(scope="module")
def second_model(tmp_path_factory):
    """A second checkpoint whose own class list is REDUCED and REORDERED relative to CONFIG's
    full 5 (Rust, Cracks, Wetspot instead of Cracks, Wetspot, Spalling, Rust, Efflorescence) — the
    case --only_classes is specifically meant to make comparable that the plain equality guard
    would otherwise reject."""
    raw = yaml.safe_load(CONFIG.read_text())
    raw["name"] = "second_model"
    raw["data"]["mvp_classes"] = ["Rust", "Cracks", "Wetspot"]
    raw["model"]["num_classes"] = 3
    raw["model"]["thin_structure_classes"] = ["Cracks"]
    raw["loss"]["per_class_alpha"] = {"Cracks": 0.35}
    raw["loss"]["per_class_gamma"] = {"Cracks": 2.5}
    tmp = tmp_path_factory.mktemp("second_model")
    config_path = tmp / "second_model.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    cfg = load_experiment_config(config_path)
    ckpt_path = tmp / "best.pt"
    torch.save({"epoch": 1, "model_state_dict": models.build_model(cfg.model).state_dict()}, ckpt_path)
    return config_path, ckpt_path


def test_only_classes_relaxes_the_shared_class_list_guard_across_differing_models(checkpoint, second_model, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    second_config, second_checkpoint = second_model
    out = tmp_path / "out"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG), str(checkpoint),
         "--model", str(second_config), str(second_checkpoint), "--coco", str(ann), "--images", str(images),
         "--only_classes", "Cracks", "Rust", "--output_dir", str(out), "--num_images", "2", "--seed", "0"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    selection = json.loads((out / "selection.json").read_text())
    assert len(selection["images"]) == 2


def test_only_classes_rejects_a_name_one_model_does_not_have(checkpoint, second_model, tmp_path):
    ann, images = _make_dataset(tmp_path / "data")
    second_config, second_checkpoint = second_model
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG), str(checkpoint),
         "--model", str(second_config), str(second_checkpoint), "--coco", str(ann), "--images", str(images),
         "--only_classes", "Spalling",  # second_model has no Spalling
         "--output_dir", str(tmp_path / "out"), "--num_images", "2"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0 and "second_model" in proc.stderr and "Spalling" in proc.stderr


def test_without_only_classes_differing_model_class_lists_are_still_rejected(checkpoint, second_model, tmp_path):
    """The relaxed guard only kicks in when --only_classes is given; without it, today's plain
    'all models must match exactly' behavior must be unchanged."""
    ann, images = _make_dataset(tmp_path / "data")
    second_config, second_checkpoint = second_model
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG), str(checkpoint),
         "--model", str(second_config), str(second_checkpoint), "--coco", str(ann), "--images", str(images),
         "--output_dir", str(tmp_path / "out"), "--num_images", "2"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0 and "same class list" in proc.stderr


# ---------------------------------------------------------------------- --sliding_window

CONFIG_NATIVE = REPO / "configs/experiments/mobilenet_native_cif_512.yaml"


@pytest.fixture(scope="module")
def checkpoint_native(tmp_path_factory):
    cfg = load_experiment_config(CONFIG_NATIVE)
    path = tmp_path_factory.mktemp("ckpt_native") / "best.pt"
    torch.save({"epoch": 3, "model_state_dict": models.build_model(cfg.model).state_dict()}, path)
    return path


def _make_large_dataset(root: Path, size: tuple[int, int] = (1200, 900)) -> tuple[Path, Path]:
    """One image well above the training tile size (512px) — the case where a plain resize/grid-tile
    and sliding-window inference can genuinely disagree (see scripts/evaluate.py's own tests)."""
    images = root / "imgs"
    images.mkdir(parents=True)
    rng = np.random.default_rng(1)
    w, h = size
    cv2.imwrite(str(images / "big.png"), rng.integers(40, 220, (h, w, 3), dtype=np.uint8))
    poly = [100, 100, 250, 100, 250, 190, 100, 190]
    coco = {"images": [{"id": 1, "file_name": "big.png", "width": w, "height": h}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [poly],
                            "bbox": [100, 100, 150, 90], "area": 150 * 90, "iscrowd": 0}],
           "categories": [{"id": 1, "name": "Cracks"}]}
    ann = root / "annotations.json"
    ann.write_text(json.dumps(coco))
    return ann, images


def test_sliding_window_renders_a_whole_image_checkpoint_at_full_resolution(checkpoint, tmp_path):
    """The user's actual ask: infer_orthophoto.py applies run_tiled_inference regardless of
    resize_mode, so visualize_predictions.py must be able to match that for a whole_image
    checkpoint too, not just render it at the squeezed evaluation resolution."""
    ann, images = _make_large_dataset(tmp_path / "data")
    proc = _run(checkpoint, tmp_path / "out", [], "--coco", ann, "--images", images, "--sliding_window", "--num_images", "1")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "sliding-window inference, full native resolution" in proc.stdout
    assert (tmp_path / "out" / "overview.png").exists()


def test_sliding_window_needs_coco_or_image_dir_not_split(checkpoint, tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG), str(checkpoint), "--sliding_window",
         "--output_dir", str(tmp_path / "out"), "--num_images", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0 and "needs --coco or --image_dir" in proc.stderr


def test_native_crop_image_dir_without_sliding_window_still_refuses(checkpoint_native, tmp_path):
    """Regression guard: the existing native_crop + --image_dir safety refusal (wrong scale if
    resized) must still fire when --sliding_window is NOT given — only the flag lifts it."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "x.png"), np.random.default_rng(0).integers(40, 220, (900, 1200, 3), dtype=np.uint8))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG_NATIVE), str(checkpoint_native),
         "--image_dir", str(img_dir), "--output_dir", str(tmp_path / "out"), "--num_images", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0 and "native_crop mode" in proc.stderr


def test_sliding_window_lifts_the_native_crop_image_dir_restriction(checkpoint_native, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "x.png"), np.random.default_rng(0).integers(40, 220, (900, 1200, 3), dtype=np.uint8))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", str(CONFIG_NATIVE), str(checkpoint_native),
         "--image_dir", str(img_dir), "--sliding_window", "--output_dir", str(tmp_path / "out"), "--num_images", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / "out" / "overview.png").exists()


def test_sliding_window_select_worst_ranks_whole_images(checkpoint, tmp_path):
    """--select worst/best must still work in --sliding_window mode (ranking whole images, not
    tiles) — exercises the do_predict()/sample_for() paths used by the ranking scan, not just the
    final render loop."""
    ann, images = _make_large_dataset(tmp_path / "data")
    proc = _run(checkpoint, tmp_path / "out", [], "--coco", ann, "--images", images,
               "--sliding_window", "--select", "worst", "--num_images", "1", "--max_scan", "1")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / "out" / "overview.png").exists()
