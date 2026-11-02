import json

from data.schema_validator import assert_disjoint_image_ids
from data.split_utils import carve_train_val, load_or_create_carve

import pytest

from configs.schema import DataConfig, DatasetSourceConfig
import data.datasets.combined_dataset as combined_dataset
from data.datasets.combined_dataset import _dacl10k_split_image_ids


def test_carve_is_pure_function_of_inputs():
    ids = list(range(1, 501))
    a = carve_train_val(ids, seed=42, val_fraction=0.1)
    b = carve_train_val(list(reversed(ids)), seed=42, val_fraction=0.1)
    assert a == b, "result must not depend on input order"
    assert len(a["val_ids"]) == 50 and len(a["train_ids"]) == 450


def test_carve_partitions_without_overlap():
    a = carve_train_val(list(range(200)), 1, 0.2)
    assert set(a["train_ids"]) | set(a["val_ids"]) == set(range(200))
    assert not set(a["train_ids"]) & set(a["val_ids"])


def test_different_seed_gives_different_split():
    assert carve_train_val(list(range(200)), 1, 0.2) != carve_train_val(list(range(200)), 2, 0.2)


def test_persisted_file_is_read_not_recomputed(tmp_path):
    path = tmp_path / "carve.json"
    first = load_or_create_carve(path, list(range(100)), 42, 0.1)
    assert path.exists()
    # different inputs/seed must be ignored because a file already exists
    second = load_or_create_carve(path, list(range(1000, 1100)), 7, 0.5)
    assert second == first
    assert json.loads(path.read_text()) == first


def test_disjoint_assertion_catches_leak():
    assert_disjoint_image_ids({1, 2}, {3, 4})
    with pytest.raises(ValueError):
        assert_disjoint_image_ids({1, 2}, {2, 3}, names=["train", "val"])


def _write_coco(path, image_ids):
    path.write_text(json.dumps({"images": [{"id": i} for i in image_ids], "annotations": [], "categories": []}))


def _dacl10k_source_cfg(tmp_path):
    train_json, val_json = tmp_path / "train.json", tmp_path / "val.json"
    _write_coco(train_json, range(1, 101))  # 100 train images
    _write_coco(val_json, range(1001, 1101))  # 100 val images
    return DatasetSourceConfig(name="dacl10k", images_roots={}, annotations={"train": str(train_json), "val": str(val_json)}, label_mapping_path="x.yaml")


def test_split_val_into_val_test_holds_out_the_configured_fraction_of_official_val(tmp_path, monkeypatch):
    """New strategy (2026-11): train uses ALL of official train (nothing carved out of it, unlike
    carve_val_from_train); official val is itself split into a kept val and a held-out test."""
    monkeypatch.setattr(combined_dataset, "REPO_ROOT", tmp_path)
    source_cfg = _dacl10k_source_cfg(tmp_path)
    data_cfg = DataConfig(dacl10k_test_strategy="split_val_into_val_test", dacl10k_test_fraction=0.2, split_seed=42)

    ids = _dacl10k_split_image_ids(source_cfg, data_cfg)

    assert set(ids["train"]) == set(range(1, 101)), "train must be ALL of official train, nothing carved out"
    assert len(ids["val"]) == 80 and len(ids["test"]) == 20, "80/20 split of the 100 official val images"
    assert set(ids["val"]) | set(ids["test"]) == set(range(1001, 1101))
    assert not set(ids["val"]) & set(ids["test"])
    persisted = tmp_path / "data" / "splits" / "dacl10k_val_test_split_seed42.json"
    assert persisted.exists(), "must persist like carve_val_from_train does, not recompute every run"


def test_split_val_into_val_test_is_rerun_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(combined_dataset, "REPO_ROOT", tmp_path)
    source_cfg = _dacl10k_source_cfg(tmp_path)
    data_cfg = DataConfig(dacl10k_test_strategy="split_val_into_val_test", dacl10k_test_fraction=0.2, split_seed=42)
    first = _dacl10k_split_image_ids(source_cfg, data_cfg)
    second = _dacl10k_split_image_ids(source_cfg, data_cfg)
    assert first == second, "second call must read the persisted file, not recompute"


def test_dacl10k_test_fraction_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        DataConfig(dacl10k_test_fraction=0.0)
    with pytest.raises(ValueError):
        DataConfig(dacl10k_test_fraction=1.0)
