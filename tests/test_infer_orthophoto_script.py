"""End-to-end test of scripts/infer_orthophoto.py as a subprocess on a synthetic orthophoto (CPU only)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import models
from configs.loader import load_experiment_config

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs/experiments/baseline_smp_mobilenet.yaml"
SCRIPT = REPO / "scripts/infer_orthophoto.py"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = load_experiment_config(CONFIG)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save({"epoch": 0, "model_state_dict": models.build_model(cfg.model).state_dict()}, path)
    return path


@pytest.fixture
def ortho(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (700, 900, 3), dtype=np.uint8)
    img[:, :150] = 0  # black no-data border on the left
    path = tmp_path / "ortho.png"
    cv2.imwrite(str(path), img)
    return path


def _run(checkpoint, ortho, out, *extra, expect_ok=True):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--checkpoint", str(checkpoint), "--image", str(ortho),
         "--output_dir", str(out), "--overlap", "64", "--batch_size", "4", *map(str, extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600,
    )
    if expect_ok:
        assert proc.returncode == 0, proc.stderr[-2000:]
    return proc


def test_full_image_run_writes_all_outputs_and_masks_no_data(checkpoint, ortho, tmp_path):
    out = tmp_path / "out"
    _run(checkpoint, ortho, out, "--scale", "0.5")
    for name in ("overlay.png", "predictions.coco.json", "run_info.json", *[f"prob_{c}.png" for c in ("Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence")]):
        assert (out / name).exists(), name
    prob = cv2.imread(str(out / "prob_Cracks.png"), cv2.IMREAD_GRAYSCALE)
    assert prob.shape == (700, 900), "outputs stay in original pixel coordinates even with --scale"
    assert prob[:, :140].max() == 0, "no-data border must be exactly zero after resampling"
    info = json.loads((out / "run_info.json").read_text())
    assert info["scale"] == 0.5 and info["crop_xywh"] is None and info["image_size"] == [900, 700]
    assert set(info["class_colors_rgb"]) == set(info["thresholds"])


def test_crop_processes_only_the_window_and_records_its_origin(checkpoint, ortho, tmp_path):
    out = tmp_path / "crop"
    _run(checkpoint, ortho, out, "--crop", 200, 100, 400, 300)
    assert cv2.imread(str(out / "prob_Rust.png"), cv2.IMREAD_GRAYSCALE).shape == (300, 400)
    info = json.loads((out / "run_info.json").read_text())
    assert info["crop_xywh"] == [200, 100, 400, 300] and info["crop_origin"] == [200, 100]
    coco = json.loads((out / "predictions.coco.json").read_text())
    assert (coco["images"][0]["width"], coco["images"][0]["height"]) == (400, 300), "COCO is in crop coordinates"


@pytest.mark.parametrize("crop", [(800, 600, 400, 400), (0, 0, 0, 100), (-5, 0, 100, 100)])
def test_bad_crop_is_rejected(checkpoint, ortho, tmp_path, crop):
    proc = _run(checkpoint, ortho, tmp_path / "bad", "--crop", *crop, expect_ok=False)
    assert proc.returncode != 0
    assert "is outside the image" in proc.stderr, proc.stderr[-500:]


def test_explicit_missing_thresholds_file_is_an_error(checkpoint, ortho, tmp_path):
    proc = _run(checkpoint, ortho, tmp_path / "t", "--thresholds", tmp_path / "nope.json", expect_ok=False)
    assert proc.returncode != 0 and "not found" in proc.stderr


def test_thresholds_file_missing_a_class_is_an_error_not_a_silent_default(checkpoint, ortho, tmp_path):
    partial = tmp_path / "thr.json"
    partial.write_text(json.dumps({"Cracks": {"threshold": 0.5, "val_dice": 0.5}}))
    proc = _run(checkpoint, ortho, tmp_path / "t2", "--thresholds", partial, expect_ok=False)
    assert proc.returncode != 0 and "Wetspot" in proc.stderr


def test_supplied_thresholds_are_used(checkpoint, ortho, tmp_path):
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({c: {"threshold": 0.9, "val_dice": 0.5} for c in ("Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence")}))
    out = tmp_path / "t3"
    _run(checkpoint, ortho, out, "--thresholds", thr)
    assert set(json.loads((out / "run_info.json").read_text())["thresholds"].values()) == {0.9}


def test_only_classes_writes_outputs_for_just_the_selected_classes(checkpoint, ortho, tmp_path):
    out = tmp_path / "filtered"
    proc = _run(checkpoint, ortho, out, "--only_classes", "Cracks", "Rust")
    assert "writing outputs only for" in proc.stdout
    for name in ("Cracks", "Rust"):
        assert (out / f"prob_{name}.png").exists()
    for name in ("Wetspot", "Spalling", "Efflorescence"):
        assert not (out / f"prob_{name}.png").exists()
    info = json.loads((out / "run_info.json").read_text())
    assert set(info["class_colors_rgb"]) == {"Cracks", "Rust"}
    assert set(info["thresholds"]) == {"Cracks", "Rust"}
    coco = json.loads((out / "predictions.coco.json").read_text())
    assert {c["name"] for c in coco["categories"]} <= {"Cracks", "Rust"}


def test_only_classes_rejects_an_unknown_name(checkpoint, ortho, tmp_path):
    proc = _run(checkpoint, ortho, tmp_path / "bad_only", "--only_classes", "Fissure", expect_ok=False)
    assert proc.returncode != 0 and "Fissure" in proc.stderr


def test_preview_is_written_only_for_large_outputs(checkpoint, ortho, tmp_path):
    small, big = tmp_path / "s", tmp_path / "b"
    _run(checkpoint, ortho, small)
    assert not (small / "overlay_preview.jpg").exists()
    _run(checkpoint, ortho, big, "--preview_max_side", 300)
    preview = cv2.imread(str(big / "overlay_preview.jpg"))
    assert max(preview.shape[:2]) == 300
