"""Rendering helpers for qualitative prediction review (pure functions, no model or dataset access).

Conventions:
- masks are boolean `(C, H, W)`, one channel per class (multi-label: they may overlap);
- overlaps stay readable because every class is drawn as a translucent fill *plus* a solid contour
  in its own colour, instead of one class hiding another;
- no-data / invalid pixels (validity mask == 0) are tinted and outlined, so it is visible which
  regions were excluded from loss and metrics rather than mistaken for "predicted nothing".
"""
from __future__ import annotations

import textwrap
from typing import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Set1-derived, one colour per MVP class in the default class order (Cracks, Wetspot, Spalling,
# Rust, Efflorescence). Cycled if a project adds classes.
PALETTE = np.array(
    [(228, 26, 28), (55, 126, 184), (152, 78, 163), (255, 127, 0), (77, 175, 74), (166, 86, 40), (247, 129, 191), (153, 153, 153)],
    dtype=np.uint8,
)
INVALID_TINT = np.array((40, 40, 70), dtype=np.float32)
INVALID_OUTLINE = (0, 255, 255)


def class_colors(n: int) -> np.ndarray:
    return PALETTE[np.arange(n) % len(PALETTE)]


def denormalize(image_chw: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Normalised `(3, H, W)` float tensor -> `(H, W, 3)` uint8 RGB."""
    img = image_chw.transpose(1, 2, 0) * np.asarray(std, np.float32) + np.asarray(mean, np.float32)
    # transpose() gives a channel-first-strided view; OpenCV rejects that layout, so make it contiguous
    return np.ascontiguousarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))


def _mark_invalid(canvas: np.ndarray, validity: np.ndarray | None) -> np.ndarray:
    if validity is None or validity.all():
        return canvas
    invalid = ~validity.astype(bool)
    out = canvas.astype(np.float32)
    out[invalid] = 0.35 * out[invalid] + 0.65 * INVALID_TINT
    out = np.ascontiguousarray(out.astype(np.uint8))
    contours, _ = cv2.findContours(invalid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, INVALID_OUTLINE, 1)
    return out


def overlay(image: np.ndarray, masks: np.ndarray, colors: np.ndarray, validity: np.ndarray | None = None, alpha: float = 0.45) -> np.ndarray:
    """`image` `(H, W, 3)` uint8, `masks` `(C, H, W)` bool -> annotated `(H, W, 3)` uint8."""
    out = image.astype(np.float32)
    for c in range(masks.shape[0]):
        m = masks[c].astype(bool)
        if validity is not None:
            m = m & validity.astype(bool)
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * colors[c].astype(np.float32)
    out = np.ascontiguousarray(out.astype(np.uint8))
    for c in range(masks.shape[0]):
        m = masks[c].astype(bool)
        if validity is not None:
            m = m & validity.astype(bool)
        if m.any():
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, tuple(int(v) for v in colors[c]), 1)
    return _mark_invalid(out, validity)


def per_class_dice(pred: np.ndarray, gt: np.ndarray, validity: np.ndarray | None = None) -> np.ndarray:
    """Dice per class on valid pixels; NaN where the class is absent from both pred and GT
    (an empty-vs-empty 1.0 would inflate scores)."""
    valid = np.ones(pred.shape[1:], bool) if validity is None else validity.astype(bool)
    p, g = pred.astype(bool) & valid, gt.astype(bool) & valid
    tp = (p & g).sum((1, 2)).astype(np.float64)
    denom = p.sum((1, 2)) + g.sum((1, 2))
    out = np.full(pred.shape[0], np.nan)
    present = denom > 0
    out[present] = 2 * tp[present] / denom[present]
    return out


def macro_dice_over_gt_classes(dice: np.ndarray, gt: np.ndarray, validity: np.ndarray | None = None) -> float:
    """Mean Dice over the classes the ground truth actually contains (NaN if it has none).
    Used to rank images as best/worst — false positives on absent classes are visible in the
    per-class numbers but do not enter the ranking, so ranking never depends on empty classes."""
    valid = np.ones(gt.shape[1:], bool) if validity is None else validity.astype(bool)
    has_gt = (gt.astype(bool) & valid).any((1, 2))
    return float(np.nanmean(dice[has_gt])) if has_gt.any() else float("nan")


def _wrap(text: str, width: int = 34) -> str:
    return textwrap.fill(text, width=width)


def _score_text(dice: np.ndarray, class_names: Sequence[str]) -> str:
    parts = [f"{n} {d:.2f}" for n, d in zip(class_names, dice) if not np.isnan(d)]
    return _wrap("   ".join(parts)) if parts else "(no classes present)"


def render_comparison(
    image: np.ndarray,
    gt: np.ndarray | None,
    predictions: dict[str, np.ndarray],
    validity: np.ndarray | None,
    class_names: Sequence[str],
    title: str,
    dpi: int = 90,
):
    """One figure: Image | Ground truth | one column per model, each an overlay of the classes.
    `gt=None` (new data without labels) drops the ground-truth column and the scores."""
    colors = class_colors(len(class_names))
    has_gt = gt is not None
    n_cols = (2 if has_gt else 1) + len(predictions)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.7 * n_cols, 5.1), dpi=dpi)
    axes = np.atleast_1d(axes)

    axes[0].imshow(_mark_invalid(image, validity))
    axes[0].set_title("Image", fontsize=10)
    if has_gt:
        axes[1].imshow(overlay(image, gt, colors, validity))
        axes[1].set_title("Ground truth", fontsize=10)
        axes[1].set_xlabel(_wrap("   ".join(n for n, m in zip(class_names, gt) if m.any())) or "(no target classes)", fontsize=8)

    for ax, (name, pred) in zip(axes[2 if has_gt else 1 :], predictions.items()):
        if has_gt:
            ax.imshow(overlay(image, pred, colors, validity))
            dice = per_class_dice(pred, gt, validity)
            macro = macro_dice_over_gt_classes(dice, gt, validity)
            ax.set_title(name if np.isnan(macro) else f"{name}\nmacro Dice {macro:.2f}", fontsize=10)
            ax.set_xlabel(_score_text(dice, class_names), fontsize=7)
        else:
            ax.imshow(overlay(image, pred, colors, validity))
            ax.set_title(name, fontsize=10)
            found = [n for n, m in zip(class_names, pred) if (m & (validity.astype(bool) if validity is not None else True)).any()]
            ax.set_xlabel(_wrap("predicted: " + ("   ".join(found) if found else "nothing")), fontsize=8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [Patch(color=colors[i] / 255.0, label=n) for i, n in enumerate(class_names)]
    if validity is not None and not validity.all():
        handles.append(Patch(facecolor=INVALID_TINT / 255.0, edgecolor="cyan", label="no-data (excluded)"))
    # legend lives in its own band below the panels so it never collides with the wrapped score text
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), max(4, n_cols * 2)), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    return fig


def stack_vertically(images: list[np.ndarray], pad: int = 6) -> np.ndarray:
    """Vertically concatenates BGR/RGB images of different widths onto a white canvas."""
    width = max(im.shape[1] for im in images)
    rows = []
    for im in images:
        canvas = np.full((im.shape[0] + pad, width, 3), 255, np.uint8)
        canvas[: im.shape[0], : im.shape[1]] = im
        rows.append(canvas)
    return np.concatenate(rows, axis=0)


def candidate_indices(concat_dataset, class_names: Sequence[str], source: str | None = None, require_classes: Sequence[str] = ()) -> list[int]:
    """Indices into a `ConcatDataset` of `CocoMultiLabelSegDataset`s, optionally restricted to one
    source and/or to images whose annotations contain every class in `require_classes`.
    Uses annotations only — no image is read."""
    wanted = [class_names.index(c) for c in require_classes]
    out: list[int] = []
    offset = 0
    for sub in concat_dataset.datasets:
        if source is None or sub.source_name == source:
            for local in range(len(sub)):
                image_id = sub.sample_image_id(local) if hasattr(sub, "sample_image_id") else sub.image_ids[local]
                if wanted:
                    present = {sub.category_id_to_channel.get(a["category_id"]) for a in sub.annotations_by_image.get(image_id, [])}
                    if not all(w in present for w in wanted):
                        continue
                out.append(offset + local)
        offset += len(sub)
    return out


def resolve_seed(seed: int | None) -> int:
    """`None` -> a fresh random seed (so every run shows different images); an int is used as given.
    The chosen value is always reported by the caller so any random draw can be reproduced later."""
    if seed is not None:
        return int(seed)
    return int(np.random.SeedSequence().generate_state(1)[0])


def pick_random(candidates: Sequence[int], n: int, seed: int) -> list[int]:
    """`n` distinct candidates, drawn without replacement; identical for identical `seed`."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(candidates), size=min(n, len(candidates)), replace=False).tolist()
