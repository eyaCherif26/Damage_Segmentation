import json

from data.schema_validator import assert_disjoint_image_ids
from data.split_utils import carve_train_val, load_or_create_carve

import pytest


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
