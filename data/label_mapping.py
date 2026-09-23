"""Loads a source-class -> MVP-class YAML mapping (see configs/data/label_maps/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
        Category ids whose mapped class is null, or not in mvp_classes, are excluded."""
        mvp_index = {name: i for i, name in enumerate(mvp_classes)}
        result: dict[int, int] = {}
        for cat_id, name in category_id_to_name.items():
            mvp_name = self.resolve(name)
            if mvp_name is None:
                continue
            if mvp_name not in mvp_index:
                raise KeyError(
                    f"Label mapping sends {name!r} -> {mvp_name!r}, but {mvp_name!r} is not in "
                    f"mvp_classes={mvp_classes}"
                )
            result[cat_id] = mvp_index[mvp_name]
        return result

    def check_covers(self, category_names: list[str]) -> None:
        """Raise if any category name in the dataset has no entry in this mapping."""
        missing = [n for n in category_names if n not in self.source_to_mvp]
        if missing:
            raise KeyError(
                f"Label mapping is missing entries for source categories: {sorted(missing)}. "
                "Every category present in the dataset must be listed explicitly, even as null."
            )
