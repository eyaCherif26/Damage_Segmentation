from pathlib import Path

import pytest
import yaml

from configs.loader import config_to_dict, load_experiment_config
from configs.schema import DataConfig, LossConfig, ModelConfig, TrainingConfig, WandbConfig

REPO = Path(__file__).resolve().parent.parent
EXPERIMENT = REPO / "configs/experiments/baseline_smp_efficientnet_b0.yaml"


def test_experiment_yaml_loads_and_resolves_sources():
    cfg = load_experiment_config(EXPERIMENT)
    assert cfg.data.image_size == 512 and cfg.model.num_classes == len(cfg.data.mvp_classes)
    assert set(cfg.data.sources_resolved) == {"cif_coco", "dacl10k"}
    assert Path(cfg.data.sources_resolved["dacl10k"].label_mapping_path).exists()
    assert "sources_resolved" in config_to_dict(cfg)["data"]


def test_unknown_yaml_key_raises(tmp_path):
    raw = yaml.safe_load(EXPERIMENT.read_text())
    raw["training"]["learnign_rate"] = 1
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(TypeError):
        load_experiment_config(p)


def test_source_without_dataset_config_path_raises(tmp_path):
    raw = yaml.safe_load(EXPERIMENT.read_text())
    raw["data"]["sources"] = ["cif_coco", "mystery"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(KeyError):
        load_experiment_config(p)


def test_resolution_floor_guard():
    with pytest.raises(ValueError):
        DataConfig(image_size=224)


def test_zero_weight_decay_guard():
    with pytest.raises(ValueError):
        TrainingConfig(weight_decay=0.0)


def test_grad_clip_max_norm_default_off_and_rejects_non_positive():
    assert TrainingConfig().grad_clip_max_norm is None, "off by default: every existing config is unaffected"
    assert TrainingConfig(grad_clip_max_norm=1.0).grad_clip_max_norm == 1.0
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            TrainingConfig(grad_clip_max_norm=bad)


def test_focal_tversky_gamma_default_off_and_rejects_non_positive():
    assert LossConfig().focal_tversky_gamma is None, "off by default: reproduces today's additive loss exactly"
    assert LossConfig(focal_tversky_gamma=2.0).focal_tversky_gamma == 2.0
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            LossConfig(focal_tversky_gamma=bad)


def test_min_lr_defaults_to_1e5_and_must_stay_below_both_peak_lrs():
    assert TrainingConfig().min_lr == 1e-5, "new global default (2026-09-29): every existing config gets this automatically"
    TrainingConfig(min_lr=1e-6, lr_head=1e-3, lr_backbone=1e-4)  # fine: below both
    with pytest.raises(ValueError):
        TrainingConfig(min_lr=0.0)
    with pytest.raises(ValueError):
        TrainingConfig(min_lr=1e-4, lr_head=1e-3, lr_backbone=1e-4)  # equal to lr_backbone, not below it
    with pytest.raises(ValueError):
        TrainingConfig(min_lr=5e-3, lr_head=1e-3, lr_backbone=1e-4)  # above lr_head entirely


def test_task_pretrained_guards():
    # every backbone may use task_pretrained (SegFormer loads a dacl10k-trained MiT, see
    # models/segformer_pretrained.py); only a missing checkpoint or unknown mode is rejected here,
    # architecture/coverage mismatches are caught at load time by the loaders themselves.
    for name in ("segformer", "smp_efficientnet_b0", "smp_mobilenet"):
        assert ModelConfig(name=name, init_mode="task_pretrained", task_pretrained_checkpoint="x.pth").init_mode == "task_pretrained"
        with pytest.raises(ValueError):
            ModelConfig(name=name, init_mode="task_pretrained")
    with pytest.raises(ValueError):
        ModelConfig(init_mode="bogus")


def test_wandb_placeholder_refuses_to_start():
    with pytest.raises(ValueError):
        WandbConfig().validate_ready()
    WandbConfig(entity="e", project="p").validate_ready()


def _raw():
    return yaml.safe_load(EXPERIMENT.read_text())


def _write(tmp_path, raw):
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_num_classes_must_match_mvp_classes(tmp_path):
    raw = _raw()
    raw["model"]["num_classes"] = 4
    with pytest.raises(ValueError, match="num_classes"):
        load_experiment_config(_write(tmp_path, raw))


def test_unknown_class_names_in_loss_or_thin_list_rejected(tmp_path):
    raw = _raw()
    raw["loss"]["per_class_alpha"] = {"Crack": 0.3}  # typo: the class is 'Cracks'
    with pytest.raises(ValueError, match="per_class_alpha"):
        load_experiment_config(_write(tmp_path, raw))
    raw = _raw()
    raw["model"]["thin_structure_classes"] = ["Hairlines"]
    with pytest.raises(ValueError, match="thin_structure_classes"):
        load_experiment_config(_write(tmp_path, raw))


def test_adding_a_class_only_needs_config_changes(tmp_path):
    raw = _raw()
    raw["data"]["mvp_classes"] = raw["data"]["mvp_classes"] + ["Weathering"]
    raw["model"]["num_classes"] = 6
    cfg = load_experiment_config(_write(tmp_path, raw))
    assert cfg.data.mvp_classes[-1] == "Weathering"


def test_removing_a_class_only_needs_config_changes(tmp_path):
    raw = _raw()
    raw["data"]["mvp_classes"] = ["Cracks", "Wetspot", "Spalling", "Rust"]  # drop Efflorescence
    raw["model"]["num_classes"] = 4
    cfg = load_experiment_config(_write(tmp_path, raw))
    assert cfg.data.mvp_classes == ["Cracks", "Wetspot", "Spalling", "Rust"]
    assert cfg.model.num_classes == 4


def test_removing_a_class_still_referenced_by_loss_or_thin_list_rejected(tmp_path):
    """Removing 'Cracks' (referenced by this config's thin_structure_classes/per_class_alpha/
    per_class_gamma) without also editing those must still fail loudly, same as a typo."""
    raw = _raw()
    raw["data"]["mvp_classes"] = ["Wetspot", "Spalling", "Rust", "Efflorescence"]
    raw["model"]["num_classes"] = 4
    with pytest.raises(ValueError, match="thin_structure_classes"):
        load_experiment_config(_write(tmp_path, raw))
