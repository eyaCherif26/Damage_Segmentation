"""Loads an experiment YAML into a fully-resolved ExperimentConfig.

Resolution is explicit, not generic-reflection-based: each nested config class has its own
`_from_dict` that only consumes the keys it knows about, so a typo'd YAML key raises immediately
via `TypeError: unexpected keyword argument` rather than being silently ignored.

Config composition is by relative-path string reference (e.g. `data.dataset_config_paths.cif_coco:
configs/data/datasets/cif_coco.yaml`), not Hydra-style deep-merge — open the experiment YAML, see
literal paths, `grep` them.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from configs.schema import (
    AugmentationConfig,
    DataConfig,
    DatasetSourceConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level of {path}, got {type(data)}")
    return data


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, raising on unknown keys instead of ignoring them."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise TypeError(f"{cls.__name__}: unknown config key(s) {sorted(unknown)}; known: {sorted(known)}")
    return cls(**data)


def load_dataset_source_config(path: Path) -> DatasetSourceConfig:
    raw = _load_yaml(path)
    base = path.parent
    raw = dict(raw)
    raw["images_roots"] = {split: str(_resolve_path(p, base)) for split, p in raw["images_roots"].items()}
    raw["annotations"] = {split: str(_resolve_path(p, base)) for split, p in raw["annotations"].items()}
    raw["label_mapping_path"] = str(_resolve_path(raw["label_mapping_path"], REPO_ROOT))
    return _build(DatasetSourceConfig, raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = _load_yaml(path)

    data_raw = dict(raw.get("data", {}))
    dataset_config_paths = data_raw.pop("dataset_config_paths", {})
    data_cfg = _build(DataConfig, data_raw)
    data_cfg.dataset_config_paths = dict(dataset_config_paths)
    resolved: dict[str, DatasetSourceConfig] = {}
    for source_name in data_cfg.sources:
        if source_name not in dataset_config_paths:
            raise KeyError(
                f"data.sources includes {source_name!r} but data.dataset_config_paths has no "
                f"entry for it. Known: {sorted(dataset_config_paths)}"
            )
        ds_path = _resolve_path(dataset_config_paths[source_name], REPO_ROOT)
        resolved[source_name] = load_dataset_source_config(ds_path)
    data_cfg.sources_resolved = resolved

    model_cfg = _build(ModelConfig, raw.get("model", {}))
    loss_cfg = _build(LossConfig, raw.get("loss", {}))
    training_cfg = _build(TrainingConfig, raw.get("training", {}))
    augmentation_cfg = _build(AugmentationConfig, raw.get("augmentation", {}))
    wandb_cfg = _build(WandbConfig, raw.get("wandb", {}))

    top_level = {k: v for k, v in raw.items() if k not in ("data", "model", "loss", "training", "augmentation", "wandb")}
    return _build(
        ExperimentConfig,
        {
            **top_level,
            "data": data_cfg,
            "model": model_cfg,
            "loss": loss_cfg,
            "training": training_cfg,
            "augmentation": augmentation_cfg,
            "wandb": wandb_cfg,
        },
    )


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """For logging (W&B config, resolved_config.yaml dump)."""
    return dataclasses.asdict(cfg)
