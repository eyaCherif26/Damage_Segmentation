"""scripts/train.py's --classes flag, run as a subprocess on CPU against real local data (the
project's own smoke-test pattern — see the script's docstring) with tiny image limits to stay fast."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
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


def test_resume_flag_errors_when_no_checkpoint_exists_yet(tmp_path):
    proc = _run(tmp_path, "--resume")
    assert proc.returncode != 0 and "no checkpoint found" in proc.stderr


def test_resume_and_resume_from_are_mutually_exclusive(tmp_path):
    proc = _run(tmp_path, "--resume", "--resume_from", str(tmp_path / "x.pt"))
    assert proc.returncode != 0 and "mutually exclusive" in proc.stderr


def test_resume_from_a_nonexistent_explicit_path_is_rejected(tmp_path):
    proc = _run(tmp_path, "--resume_from", str(tmp_path / "nope.pt"))
    assert proc.returncode != 0 and "does not exist" in proc.stderr


def test_resume_continues_training_from_the_next_epoch_without_rerunning_completed_ones(tmp_path):
    first = _run(tmp_path, "--max_epochs", "2")
    assert first.returncode == 0, first.stderr[-3000:]
    ckpt_dir = tmp_path / "baseline_smp_efficientnet_b0"
    state = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 1  # 0-indexed: epochs 0 and 1 both ran
    assert "scaler_state_dict" in state and "early_stopping_state" in state

    second = _run(tmp_path, "--max_epochs", "3", "--resume")
    assert second.returncode == 0, second.stderr[-3000:]
    assert "continuing from epoch 2" in second.stdout
    assert "[epoch 0]" not in second.stdout and "[epoch 1]" not in second.stdout, "must not re-run already-completed epochs"
    assert "[epoch 2]" in second.stdout
    state2 = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state2["epoch"] == 2


def test_resume_with_a_changed_max_epochs_warns_about_the_lr_schedule(tmp_path):
    first = _run(tmp_path, "--max_epochs", "1")
    assert first.returncode == 0, first.stderr[-3000:]
    second = _run(tmp_path, "--max_epochs", "5", "--resume")
    assert second.returncode == 0, second.stderr[-3000:]
    assert "max_epochs=5" in second.stdout and "max_epochs=1" in second.stdout


def test_resume_when_already_at_max_epochs_is_a_safe_no_op(tmp_path):
    first = _run(tmp_path, "--max_epochs", "1")
    assert first.returncode == 0, first.stderr[-3000:]
    second = _run(tmp_path, "--max_epochs", "1", "--resume")
    assert second.returncode == 0, second.stderr[-3000:]
    assert "already reached" in second.stdout
    assert "[epoch 0]" not in second.stdout


# ------------------------------------------------- fine-tuning from a prior run's own checkpoint

SEGFORMER_CONFIG = REPO / "configs/experiments/baseline_segformer.yaml"


def test_finetune_from_a_prior_checkpoint_transfers_encoder_and_reinits_classifier(tmp_path):
    """The actual fine-tuning use case: point model.task_pretrained_checkpoint at THIS project's own
    prior run (not an external checkpoint), targeting a different number of classes, with a
    head-only freeze_schedule step -- after training, the encoder must be byte-identical to the
    transferred source (nothing but the head had requires_grad=True) and the classifier must have
    the NEW run's class count, not the source's."""
    import torch

    import models
    from configs.schema import ModelConfig

    source_model = models.build_model(ModelConfig(name="segformer", encoder_name="nvidia/mit-b0", num_classes=3, dropout=0.2))
    ckpt_path = tmp_path / "prior_run_best.pt"
    torch.save({"epoch": 9, "model_state_dict": source_model.state_dict()}, ckpt_path)

    cfg = yaml.safe_load(SEGFORMER_CONFIG.read_text())
    cfg["data"]["mvp_classes"] = ["Cracks", "Wetspot", "Spalling", "Rust"]  # 4 classes: different from source's 3
    cfg["model"]["num_classes"] = 4
    cfg["model"]["init_mode"] = "task_pretrained"
    cfg["model"]["task_pretrained_checkpoint"] = str(ckpt_path)
    cfg["model"]["load_decode_head"] = True
    cfg["model"]["load_classifier"] = False
    cfg["training"]["frozen_epochs"] = 0  # mutually exclusive with freeze_schedule
    cfg["training"]["freeze_schedule"] = [{"epoch": 0, "unfrozen": ["head"]}]  # nothing in the backbone should move
    finetune_config = tmp_path / "finetune.yaml"
    finetune_config.write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NO_ALBUMENTATIONS_UPDATE": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(finetune_config), "--wandb_mode", "disabled",
         "--max_epochs", "1", "--max_train_images", "4", "--max_val_images", "4",
         "--checkpoint_dir", str(tmp_path / "run")],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "classifier_reinitialised': True" in proc.stdout, proc.stdout[-3000:]
    assert "[epoch 0] freeze_schedule step: unfrozen=['head']" in proc.stdout

    result = torch.load(tmp_path / "run" / "baseline_segformer" / "last.pt", map_location="cpu", weights_only=False)
    trained_state = result["model_state_dict"]
    source_state = source_model.state_dict()
    for k, v in source_state.items():
        if k.startswith("net.segformer."):
            assert torch.equal(trained_state[k], v), f"{k} moved despite the backbone being frozen all epoch"
    assert trained_state["net.decode_head.classifier.weight"].shape[0] == 4, "classifier must match the NEW run's class count"
