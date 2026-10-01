from pathlib import Path

import pytest
import yaml

from data.label_mapping import LabelMapping

REPO = Path(__file__).resolve().parent.parent
MVP = ["Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence"]


def _native(dataset_yaml: str) -> list[str]:
    return yaml.safe_load((REPO / "configs/data/datasets" / dataset_yaml).read_text())["native_categories"]


@pytest.mark.parametrize(
    "mapping_file,dataset_yaml",
    [("dacl10k_19_to_mvp5.yaml", "dacl10k.yaml"), ("cif6_to_mvp5.yaml", "cif_coco.yaml")],
)
def test_mapping_covers_every_native_category(mapping_file, dataset_yaml):
    mapping = LabelMapping.load(REPO / "configs/data/label_maps" / mapping_file)
    mapping.check_covers(_native(dataset_yaml))
    assert set(mapping.source_to_mvp) == set(_native(dataset_yaml)), "mapping has entries for classes the dataset doesn't have"


@pytest.mark.parametrize("mapping_file", ["dacl10k_19_to_mvp5.yaml", "cif6_to_mvp5.yaml"])
def test_all_targets_are_mvp_classes_or_null(mapping_file):
    mapping = LabelMapping.load(REPO / "configs/data/label_maps" / mapping_file)
    for src, tgt in mapping.source_to_mvp.items():
        assert tgt is None or tgt in MVP, f"{src} -> {tgt}"


def test_verified_mappings_from_existing_user_data():
    dacl = LabelMapping.load(REPO / "configs/data/label_maps/dacl10k_19_to_mvp5.yaml")
    assert dacl.resolve("Weathering") == "Wetspot"
    assert dacl.resolve("ACrack") == "Cracks"
    assert dacl.resolve("Cavity") == "Spalling"
    assert dacl.resolve("Hollowareas") is None
    cif = LabelMapping.load(REPO / "configs/data/label_maps/cif6_to_mvp5.yaml")
    assert cif.resolve("CrackPrecip") == "Efflorescence"
    assert cif.resolve("Algae") == "Wetspot"
    assert cif.resolve("NetCrack") == "Cracks"


def test_unknown_source_class_raises():
    mapping = LabelMapping({"A": "Cracks"})
    with pytest.raises(KeyError):
        mapping.resolve("B")
    with pytest.raises(KeyError):
        mapping.check_covers(["A", "B"])


def test_channel_mapping_skips_null_and_orders_by_mvp_list():
    mapping = LabelMapping({"Crack": "Cracks", "Graffiti": None, "Rust": "Rust"})
    out = mapping.build_category_id_to_channel({1: "Crack", 2: "Graffiti", 3: "Rust"}, MVP)
    assert out == {1: 0, 3: 3}


def test_validate_targets_rejects_typo():
    """A target that isn't a real class name at all — almost always a label-map typo — still
    raises, via validate_targets() (moved off build_category_id_to_channel, see next test)."""
    mapping = LabelMapping({"X": "NotAClass"})
    with pytest.raises(ValueError, match="NotAClass"):
        mapping.validate_targets(MVP)


def test_build_category_id_to_channel_excludes_valid_but_unselected_class():
    """A target that IS a real class, just not selected for this run's (possibly reduced)
    mvp_classes, is excluded like a null mapping — not an error. This is what makes training on a
    class subset possible: a label map written against the full taxonomy still works unmodified
    against a run that only selects some of those classes."""
    mapping = LabelMapping({"X": "Efflorescence"})
    out = mapping.build_category_id_to_channel({1: "X"}, ["Cracks", "Rust"])  # Efflorescence not selected
    assert out == {}
