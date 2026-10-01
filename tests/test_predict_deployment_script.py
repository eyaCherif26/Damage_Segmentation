"""scripts/predict_deployment.py, run as a subprocess on CPU against a bundle from
scripts/export_deployment_model.py — no experiment YAML involved at predict time."""
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
EXPORT_SCRIPT = REPO / "scripts/export_deployment_model.py"
PREDICT_SCRIPT = REPO / "scripts/predict_deployment.py"
CLASSES = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}


def _run(*args, timeout=180):
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=ENV, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """The checkpoint + thresholds file the bundle is exported from — kept around so tests can
    also drive scripts/infer_orthophoto.py (the normal config-based path) on the exact same
    weights, for a genuine cross-path comparison rather than just 'doesn't crash'."""
    cfg = load_experiment_config(CONFIG)
    tmp = tmp_path_factory.mktemp("source")
    ckpt = tmp / "best.pt"
    torch.save({"epoch": 5, "model_state_dict": models.build_model(cfg.model).state_dict()}, ckpt)
    thr = tmp / "thresholds.json"
    thr.write_text(json.dumps({c: {"threshold": 0.5, "val_dice": 0.5} for c in CLASSES}))
    return {"checkpoint": ckpt, "thresholds": thr}


@pytest.fixture(scope="module")
def bundle(source, tmp_path_factory):
    """A real export, done once via the actual export script (not hand-built), reused read-only
    by every test below — exercises the real export -> predict boundary, not a shortcut."""
    out = tmp_path_factory.mktemp("bundle") / "bundle"
    proc = _run(EXPORT_SCRIPT, "--config", CONFIG, "--checkpoint", source["checkpoint"], "--output_dir", out,
               "--thresholds", source["thresholds"])
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


@pytest.fixture
def scene(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (600, 800, 3), dtype=np.uint8)
    img[:, :120] = 0  # black no-data border
    path = tmp_path / "scene.png"
    cv2.imwrite(str(path), img)
    return path


def test_predict_uses_the_bundled_thresholds_by_default(bundle, scene, tmp_path):
    out = tmp_path / "out"
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    coco = json.loads((out / "predictions.coco.json").read_text())
    assert coco["images"] == [{"id": 1, "width": 800, "height": 600}]
    assert {c["name"] for c in coco["categories"]} == set(CLASSES)
    assert "thresholds from" in proc.stdout and str(bundle / "thresholds.json") in proc.stdout


def test_predict_fixed_threshold_overrides_the_bundle(bundle, scene, tmp_path):
    out = tmp_path / "out"
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", out, "--threshold", "0.9")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "fixed threshold 0.9 for every class" in proc.stdout


def test_predict_threshold_out_of_range_is_rejected(bundle, scene, tmp_path):
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", tmp_path / "out", "--threshold", "1.5")
    assert proc.returncode != 0 and "must be in [0, 1]" in proc.stderr


def test_predict_explicit_missing_thresholds_file_is_an_error(bundle, scene, tmp_path):
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", tmp_path / "out",
               "--thresholds", tmp_path / "nope.json")
    assert proc.returncode != 0 and "does not exist" in proc.stderr


def test_predict_crop_processes_only_the_window(bundle, scene, tmp_path):
    out = tmp_path / "out"
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", out, "--crop", 100, 50, 300, 200)
    assert proc.returncode == 0, proc.stderr[-2000:]
    coco = json.loads((out / "predictions.coco.json").read_text())
    assert (coco["images"][0]["width"], coco["images"][0]["height"]) == (300, 200)
    assert "crop coordinates" in proc.stdout


def test_predict_overlay_flag_writes_an_extra_png(bundle, scene, tmp_path):
    without = tmp_path / "without"
    _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", without)
    assert not (without / "overlay.png").exists()

    with_overlay = tmp_path / "with"
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", with_overlay, "--overlay")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (with_overlay / "overlay.png").exists()


def test_predict_matches_infer_orthophoto_numerically(bundle, source, scene, tmp_path):
    """The deployment path (pickled model, no config) must give the SAME predictions as the
    normal config-based path (scripts/infer_orthophoto.py) on the same weights/image/thresholds —
    not just 'doesn't crash'. Both ultimately write a predictions.coco.json via the same
    inference/postprocess.py::export_coco_predictions, so the annotation lists are directly comparable."""
    deployment_out = tmp_path / "deployment"
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", deployment_out)
    assert proc.returncode == 0, proc.stderr[-2000:]

    infer_ortho_out = tmp_path / "infer_ortho"
    infer_script = REPO / "scripts/infer_orthophoto.py"
    proc2 = _run(infer_script, "--config", CONFIG, "--checkpoint", source["checkpoint"], "--image", scene,
                "--output_dir", infer_ortho_out, "--thresholds", source["thresholds"])
    assert proc2.returncode == 0, proc2.stderr[-2000:]

    deployment_coco = json.loads((deployment_out / "predictions.coco.json").read_text())
    infer_ortho_coco = json.loads((infer_ortho_out / "predictions.coco.json").read_text())
    assert deployment_coco["annotations"] == infer_ortho_coco["annotations"]
    assert deployment_coco["categories"] == infer_ortho_coco["categories"]


def test_only_classes_writes_a_subset_and_still_matches_the_config_based_path(bundle, source, scene, tmp_path):
    out = tmp_path / "out"
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", out,
               "--only_classes", "Cracks", "Rust")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "writing outputs only for ['Cracks', 'Rust']" in proc.stdout
    coco = json.loads((out / "predictions.coco.json").read_text())
    assert {c["name"] for c in coco["categories"]} == {"Cracks", "Rust"}

    infer_ortho_out = tmp_path / "infer_ortho"
    proc2 = _run(REPO / "scripts/infer_orthophoto.py", "--config", CONFIG, "--checkpoint", source["checkpoint"],
                "--image", scene, "--output_dir", infer_ortho_out, "--thresholds", source["thresholds"],
                "--only_classes", "Cracks", "Rust")
    assert proc2.returncode == 0, proc2.stderr[-2000:]
    infer_ortho_coco = json.loads((infer_ortho_out / "predictions.coco.json").read_text())
    assert coco["annotations"] == infer_ortho_coco["annotations"]


def test_only_classes_rejects_an_unknown_name(bundle, scene, tmp_path):
    proc = _run(PREDICT_SCRIPT, "--bundle", bundle, "--image", scene, "--output_dir", tmp_path / "out",
               "--only_classes", "Fissure")
    assert proc.returncode != 0 and "Fissure" in proc.stderr


def test_segformer_bundle_survives_a_fresh_process_round_trip(tmp_path_factory):
    """Regression test for a real bug found during development: transformers >= 5.x's SegFormer
    encoder captures its internal per-stage hidden states via forward hooks installed lazily,
    gated by a process-global registry populated only inside PreTrainedModel.__init__. A plain
    torch.load() never calls __init__ (pickling restores __dict__ directly), so in a FRESH process
    that only ever unpickles (never constructs a SegFormer model normally first), the very first
    forward pass used to silently get None where it expected captured hidden states and crash.
    This only reproduces in a genuinely separate process — reusing pytest's own process would be
    masked by earlier tests having already constructed a SegformerModel normally, populating the
    registry as a side effect. Hence: export AND predict as two fresh subprocesses here."""
    config = REPO / "configs/experiments/baseline_segformer.yaml"  # MiT-b0, from_scratch: small & fast
    cfg = load_experiment_config(config)
    tmp = tmp_path_factory.mktemp("segformer_pickle_regression")
    ckpt = tmp / "best.pt"
    torch.save({"epoch": 0, "model_state_dict": models.build_model(cfg.model).state_dict()}, ckpt)
    bundle_dir = tmp / "bundle"
    export_proc = _run(EXPORT_SCRIPT, "--config", config, "--checkpoint", ckpt, "--output_dir", bundle_dir, timeout=120)
    assert export_proc.returncode == 0, export_proc.stderr[-2000:]

    rng = np.random.default_rng(0)
    scene_path = tmp / "scene.png"
    cv2.imwrite(str(scene_path), rng.integers(60, 200, (600, 800, 3), dtype=np.uint8))
    out = tmp / "out"
    predict_proc = _run(PREDICT_SCRIPT, "--bundle", bundle_dir, "--image", scene_path, "--output_dir", out,
                        "--threshold", "0.5", timeout=120)
    assert predict_proc.returncode == 0, predict_proc.stderr[-3000:]
    coco = json.loads((out / "predictions.coco.json").read_text())
    assert coco["images"][0]["width"] == 800 and coco["images"][0]["height"] == 600
