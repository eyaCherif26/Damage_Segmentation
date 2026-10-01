"""scripts/train.py's --classes flag, run as a subprocess on CPU against real local data (the
project's own smoke-test pattern — see the script's docstring) with tiny image limits to stay fast."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts/train.py"
CONFIG = REPO / "configs/experiments/baseline_smp_efficientnet_b0.yaml"


def _run(tmp_path, *extra):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--wandb_mode", "disabled",
         "--max_epochs", "1", "--max_train_images", "4", "--max_val_images", "4",
         "--checkpoint_dir", str(tmp_path), *map(str, extra)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600,
    )


def test_classes_flag_trains_on_a_reduced_subset(tmp_path):
    proc = _run(tmp_path, "--classes", "Cracks", "Rust")
    assert proc.returncode == 0, proc.stderr[-3000:]
    resolved = yaml.safe_load((tmp_path / "baseline_smp_efficientnet_b0" / "resolved_config.yaml").read_text())
    assert resolved["data"]["mvp_classes"] == ["Cracks", "Rust"]
    assert resolved["model"]["num_classes"] == 2


def test_classes_flag_rejects_a_class_still_referenced_by_the_loss_config(tmp_path):
    """baseline_smp_efficientnet_b0.yaml's thin_structure_classes/per_class_alpha/per_class_gamma
    all reference 'Cracks' — dropping it via --classes without also editing those must fail
    loudly (before any dataset/training work starts), not silently ignore the stale reference."""
    proc = _run(tmp_path, "--classes", "Wetspot", "Spalling", "Rust", "Efflorescence")
    assert proc.returncode != 0
    assert "thin_structure_classes" in proc.stderr
    assert not (tmp_path / "baseline_smp_efficientnet_b0").exists(), "must fail before any training I/O"


def test_lr_flag_sets_both_head_and_backbone_to_one_shared_value(tmp_path):
    proc = _run(tmp_path, "--lr", "0.0003")
    assert proc.returncode == 0, proc.stderr[-3000:]
    resolved = yaml.safe_load((tmp_path / "baseline_smp_efficientnet_b0" / "resolved_config.yaml").read_text())
    assert resolved["training"]["lr_head"] == 0.0003 and resolved["training"]["lr_backbone"] == 0.0003


def test_lr_head_and_lr_backbone_flags_set_them_individually(tmp_path):
    proc = _run(tmp_path, "--lr_head", "0.002", "--lr_backbone", "0.0002")
    assert proc.returncode == 0, proc.stderr[-3000:]
    resolved = yaml.safe_load((tmp_path / "baseline_smp_efficientnet_b0" / "resolved_config.yaml").read_text())
    assert resolved["training"]["lr_head"] == 0.002 and resolved["training"]["lr_backbone"] == 0.0002


def test_individual_lr_flags_override_the_shared_lr_flag(tmp_path):
    """--lr_head/--lr_backbone must win over --lr when both are given, not silently be ignored."""
    proc = _run(tmp_path, "--lr", "0.0003", "--lr_head", "0.002")
    assert proc.returncode == 0, proc.stderr[-3000:]
    resolved = yaml.safe_load((tmp_path / "baseline_smp_efficientnet_b0" / "resolved_config.yaml").read_text())
    assert resolved["training"]["lr_head"] == 0.002 and resolved["training"]["lr_backbone"] == 0.0003


def test_lr_flags_below_min_lr_are_rejected_before_any_training_io(tmp_path):
    """min_lr defaults to 1e-5 — setting lr_backbone at or below that must still raise (the
    schema's own guard), not silently produce a scheduler that never anneals correctly."""
    proc = _run(tmp_path, "--lr_backbone", "0.000005")
    assert proc.returncode != 0 and "min_lr" in proc.stderr
    assert not (tmp_path / "baseline_smp_efficientnet_b0").exists()
