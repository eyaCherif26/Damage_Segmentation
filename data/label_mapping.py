"""Loads a source-class -> MVP-class YAML mapping (see configs/data/label_maps/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import yaml


@dataclass
class LabelMapping:
    source_to_mvp: dict[str, str | None]  # source class name -> MVP class name, or None if dropped

    @classmethod
    def load(cls, path: str | Path) -> "LabelMapping":
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping at {path}, got {type(raw)}")
        return cls(source_to_mvp=raw)

    @classmethod
    def identity_for(cls, category_names: list[str], mvp_classes: list[str]) -> "LabelMapping":
        """For a dataset whose categories already ARE the MVP class names (e.g. the 5-class
        remapped COCO files). Any other category raises — the caller must supply a label map (which
        may send it to null) instead of having it silently dropped."""
        unknown = [n for n in category_names if n not in mvp_classes]
        if unknown:
            raise ValueError(
                f"Categories {sorted(unknown)} are not MVP classes {mvp_classes}. Pass a label map YAML that maps "
                "every category (use null for the ones to ignore); nothing is dropped silently."
            )
        return cls(source_to_mvp={n: n for n in category_names})

    def resolve(self, source_class: str) -> str | None:
        if source_class not in self.source_to_mvp:
            raise KeyError(
                f"Source class {source_class!r} is not present in this label mapping "
                f"(known: {sorted(self.source_to_mvp)}). Every category name in the source "
                "dataset must have an explicit entry (even if it maps to null) — this is "
                "deliberate so a missing/typo'd class fails loudly rather than being silently "
                "dropped."
            )
        return self.source_to_mvp[source_class]

    def build_category_id_to_channel(
        self, category_id_to_name: dict[int, str], mvp_classes: list[str]
    ) -> dict[int, int]:
        """category_id (as used in a COCO file) -> channel index into the mvp_classes list.
        Category ids whose mapped class is null, or not in mvp_classes (e.g. a class that exists
        in this mapping but was deliberately left out of THIS run's mvp_classes subset), are
        excluded — treated the same as a confirmed negative. This does NOT catch a typo'd target
        (a mapping to a class that isn't a real class name at all); call validate_targets() first
        against the canonical class taxonomy for that."""
        mvp_index = {name: i for i, name in enumerate(mvp_classes)}
        result: dict[int, int] = {}
        for cat_id, name in category_id_to_name.items():
            mvp_name = self.resolve(name)
            if mvp_name is None or mvp_name not in mvp_index:
                continue
            result[cat_id] = mvp_index[mvp_name]
        return result

    def validate_targets(self, canonical_classes: Collection[str]) -> None:
        """Raise if any non-null mapping target is not a member of `canonical_classes` — the full
        class taxonomy this label map is meant to target, independent of any one run's *selected*
        subset (that's what build_category_id_to_channel's mvp_classes handles). This is the typo
        check: a target that's a real class name just not chosen for the current run is fine (the
        caller should pass the union of the fixed taxonomy and the run's own mvp_classes so a
        legitimately-added class validates too); a target that isn't a real class name at all —
        almost always a typo in the label-map YAML — is not."""
        unknown = sorted({v for v in self.source_to_mvp.values() if v is not None} - set(canonical_classes))
        if unknown:
            raise ValueError(
                f"Label mapping targets {unknown} are not members of the canonical class taxonomy "
                f"{sorted(canonical_classes)}. This is usually a typo in the label-map YAML — if you "
                "meant to train on a reduced class subset, that's controlled by data.mvp_classes, "
                "not the label map."
            )

    def check_covers(self, category_names: list[str]) -> None:
        """Raise if any category name in the dataset has no entry in this mapping."""
        missing = [n for n in category_names if n not in self.source_to_mvp]
        if missing:
            raise KeyError(
                f"Label mapping is missing entries for source categories: {sorted(missing)}. "
                "Every category present in the dataset must be listed explicitly, even as null."
            )
