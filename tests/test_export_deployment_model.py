"""scripts/export_deployment_model.py, run as a subprocess on CPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import models
from configs.loader import load_experiment_config

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs/experiments/baseline_smp_mobilenet.yaml"
SCRIPT = REPO / "scripts/export_deployment_model.py"
CLASSES = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = load_experiment_config(CONFIG)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save({"epoch": 7, "model_state_dict": models.build_model(cfg.model).state_dict()}, path)
    return path


def _run(checkpoint, out, *extra):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--checkpoint", str(checkpoint),
         "--output_dir", str(out), *map(str, extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )


def test_export_writes_a_loadable_model_plus_classes_and_meta(checkpoint, tmp_path):
    out = tmp_path / "bundle"
    proc = _run(checkpoint, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (out / "model.pt").exists()

    classes = json.loads((out / "classes.json").read_text())
    assert classes == [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]

    meta = json.loads((out / "meta.json").read_text())
    assert meta["tile_size"] == 512 and meta["resize_mode"] == "whole_image" and meta["pixel_scale"] == 1.0
    assert meta["source_epoch"] == 7

    # the exported object is a real, standalone model — loadable with nothing else from this repo's configs
    model = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 512, 512))
    assert y.shape == (1, len(CLASSES), 512, 512)


def test_export_skips_thresholds_when_none_exist(checkpoint, tmp_path):
    out = tmp_path / "bundle_no_thr"
    proc = _run(checkpoint, out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not (out / "thresholds.json").exists()
    assert "no thresholds file" in proc.stdout


def test_export_bundles_an_explicit_thresholds_file(checkpoint, tmp_path):
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({c: {"threshold": 0.3, "val_dice": 0.5} for c in CLASSES}))
    out = tmp_path / "bundle_thr"
    proc = _run(checkpoint, out, "--thresholds", thr)
    assert proc.returncode == 0, proc.stderr[-2000:]
    bundled = json.loads((out / "thresholds.json").read_text())
    assert all(bundled[c]["threshold"] == pytest.approx(0.3) for c in CLASSES)


def test_export_rejects_hand_broken_json_thresholds_file(checkpoint, tmp_path):
    """A thresholds.json hand-edited with '//' comments (JSON has no comment syntax) must fail
    loudly at export time, not silently ship a broken file into the deployment bundle."""
    thr = tmp_path / "thr.json"
    thr.write_text("// {\n//   \"Cracks\": 0.3\n// }\n\n{\"Cracks\": {\"threshold\": 0.3}}")
    out = tmp_path / "bundle_bad_thr"
    proc = _run(checkpoint, out, "--thresholds", thr)
    assert proc.returncode != 0 and "not valid JSON" in proc.stderr
    assert not (out / "thresholds.json").exists()
