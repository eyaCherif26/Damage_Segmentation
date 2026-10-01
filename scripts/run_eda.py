"""EDA for multi-label segmentation under the unified MVP taxonomy.

    python scripts/run_eda.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml --output_dir docs/eda

Uses the same train/val/test role assignment as training (incl. the persisted dacl10k carve), so the
numbers describe exactly what is trained/evaluated on. Writes eda_summary.json, eda_report.md and PNGs.

Pixel statistics are computed on 256x256 rasterizations of the annotations (polygon coordinates are
scaled, images are not read), which is exact enough for class shares/co-occurrence and fast.
Physical sizes (mm) are NOT reported: the GSD of the source datasets is unknown, so only pixels
(native and at the evaluation resolution) and image-area fractions are given.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils

from configs.loader import load_experiment_config
from data.common_schema import decode_segmentation
from data.datasets.combined_dataset import build_split_datasets

GRID = 256
PCTS = (5, 25, 50, 75, 95)


def rasterize_scaled(anns: list[dict], w: int, h: int, cat_to_ch: dict[int, int], n_ch: int) -> np.ndarray:
    """(C, GRID, GRID) bool. Polygons are scaled to the grid before rasterization (no full-size decode)."""
    out = np.zeros((n_ch, GRID, GRID), dtype=bool)
    sx, sy = GRID / w, GRID / h
    for a in anns:
        ch = cat_to_ch.get(a["category_id"])
        if ch is None:
            continue
        seg = a["segmentation"]
        if isinstance(seg, list):
            polys = [[c * (sx if i % 2 == 0 else sy) for i, c in enumerate(p)] for p in seg if len(p) >= 6]
            if not polys:
                continue
            m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polys, GRID, GRID))).astype(bool)
        else:
            m = cv2.resize(decode_segmentation(seg, h, w), (GRID, GRID), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[ch] |= m
    return out


def pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{p}": float("nan") for p in PCTS}
    q = np.percentile(values, PCTS)
    return {f"p{p}": float(v) for p, v in zip(PCTS, q)}


def analyse(ds, classes: list[str]) -> dict:
    n_ch = len(classes)
    px = np.zeros(n_ch)
    img_has = np.zeros(n_ch, int)
    co_px = np.zeros((n_ch, n_ch))
    co_img = np.zeros((n_ch, n_ch), int)
    inst = np.zeros(n_ch, int)
    area_native: dict[int, list[float]] = defaultdict(list)
    area_frac: dict[int, list[float]] = defaultdict(list)
    aspect: dict[int, list[float]] = defaultdict(list)
    n_img_any, sizes = 0, defaultdict(int)

    for image_id in ds.image_ids:
        info = ds.images_by_id[image_id]
        w, h = info["width"], info["height"]
        sizes[f"{w}x{h}"] += 1
        anns = ds.annotations_by_image.get(image_id, [])
        for a in anns:
            ch = ds.category_id_to_channel.get(a["category_id"])
            if ch is None:
                continue
            inst[ch] += 1
            area_native[ch].append(a["area"])
            area_frac[ch].append(a["area"] / (w * h))
            bw, bh = a["bbox"][2], a["bbox"][3]
            aspect[ch].append(max(bw, bh) / max(1.0, min(bw, bh)))
        m = rasterize_scaled(anns, w, h, ds.category_id_to_channel, n_ch)
        present = m.reshape(n_ch, -1).any(1)
        n_img_any += int(present.any())
        px += m.reshape(n_ch, -1).sum(1) / (GRID * GRID)
        img_has += present
        flat = m.reshape(n_ch, -1).astype(np.float32)
        co_px += flat @ flat.T / (GRID * GRID)
        co_img += np.outer(present, present).astype(int)

    n = len(ds.image_ids)
    eval_res = 512
    return {
        "n_images": n, "n_images_with_any_target": n_img_any,
        "instances": {c: int(inst[i]) for i, c in enumerate(classes)},
        "image_fraction_with_class": {c: float(img_has[i] / max(1, n)) for i, c in enumerate(classes)},
        "pixel_share_of_all_pixels": {c: float(px[i] / max(1, n)) for i, c in enumerate(classes)},
        "instance_area_native_px2": {c: pct(area_native[i]) for i, c in enumerate(classes)},
        "instance_area_fraction_of_image": {c: pct(area_frac[i]) for i, c in enumerate(classes)},
        "instance_char_size_at_eval_res_px": {c: pct([float(np.sqrt(f) * eval_res) for f in area_frac[i]]) for i, c in enumerate(classes)},
        "instance_bbox_aspect": {c: pct(aspect[i]) for i, c in enumerate(classes)},
        "pixel_cooccurrence_P_b_given_a": (co_px / np.maximum(np.diag(co_px)[:, None], 1e-9)).round(4).tolist(),
        "image_cooccurrence_counts": co_img.tolist(),
        "resolutions": dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:8]), "n_distinct_resolutions": len(sizes),
    }


def image_stats(ds, k: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = rng.choice(ds.image_ids, size=min(k, len(ds.image_ids)), replace=False)
    means, stds, dark, bright = [], [], [], []
    for i in ids:
        img = ds._load_image(ds.images_by_id[int(i)]["file_name"])
        small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        means.append(small.mean((0, 1)))
        stds.append(small.std((0, 1)))
        gray = small.mean(2)
        dark.append(float((gray < 0.05).mean()))
        bright.append(float(gray.mean()))
    return {
        "n_sampled": len(ids), "rgb_mean": np.mean(means, 0).round(4).tolist(), "rgb_std_within_image": np.mean(stds, 0).round(4).tolist(),
        "brightness_mean_p5_p50_p95": np.percentile(bright, [5, 50, 95]).round(4).tolist(),
        "frac_images_with_gt5pct_near_black_pixels": float(np.mean(np.array(dark) > 0.05)),
    }


def sample_grid(ds, classes: list[str], path: Path, n: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cand = [i for i in ds.image_ids if any(ds.category_id_to_channel.get(a["category_id"]) is not None for a in ds.annotations_by_image.get(i, []))]
    ids = rng.choice(cand, size=min(n, len(cand)), replace=False)
    colors = plt.cm.tab10(np.arange(len(classes)))[:, :3]
    fig, axes = plt.subplots(2, len(ids) // 2, figsize=(4 * (len(ids) // 2), 8))
    for ax, i in zip(axes.ravel(), ids):
        info = ds.images_by_id[int(i)]
        img = cv2.resize(ds._load_image(info["file_name"]), (384, 384)).astype(np.float32) / 255
        m = rasterize_scaled(ds.annotations_by_image.get(int(i), []), info["width"], info["height"], ds.category_id_to_channel, len(classes))
        m = cv2.resize(m.transpose(1, 2, 0).astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST).astype(bool)
        for c in range(len(classes)):
            img[m[..., c]] = 0.55 * img[m[..., c]] + 0.45 * colors[c]
        ax.imshow(img)
        ax.set_title(", ".join(classes[c] for c in range(len(classes)) if m[..., c].any()) or "(none)", fontsize=8)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in range(len(classes))]
    fig.legend(handles, classes, loc="lower center", ncol=len(classes))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--image_stat_samples", type=int, default=300)
    p.add_argument("--skip_images", action="store_true", help="Skip anything that reads image files (stats, sample grids).")
    args = p.parse_args()

    cfg = load_experiment_config(args.config)
    classes = cfg.data.mvp_classes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.resize_mode = "whole_image"  # EDA describes annotations per image, whatever the training crop mode is
    datasets = build_split_datasets(cfg.data, None, None)  # transforms unused: annotations only

    summary: dict = {"classes": classes, "dacl10k_test_strategy": cfg.data.dacl10k_test_strategy, "splits": {}}
    for split, concat in datasets.items():
        for src_ds in concat.datasets:
            key = f"{src_ds.source_name}/{split}"
            print(f"[eda] {key}: {len(src_ds)} images")
            summary["splits"][key] = analyse(src_ds, classes)

    if not args.skip_images:
        train_sets = {d.source_name: d for d in datasets["train"].datasets}
        summary["image_stats_train"] = {}
        for name, ds in train_sets.items():
            print(f"[eda] image stats + samples: {name}")
            summary["image_stats_train"][name] = image_stats(ds, args.image_stat_samples)
            sample_grid(ds, classes, args.output_dir / f"samples_{name}.png")

    (args.output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    write_plots_and_report(summary, classes, args.output_dir)
    print(f"[eda] wrote {args.output_dir}/eda_summary.json, eda_report.md and figures")


def write_plots_and_report(summary: dict, classes: list[str], out: Path) -> None:
    keys = list(summary["splits"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    w = 0.8 / len(keys)
    for j, k in enumerate(keys):
        s = summary["splits"][k]
        axes[0].bar(np.arange(len(classes)) + j * w, [s["instances"][c] for c in classes], w, label=k)
        axes[1].bar(np.arange(len(classes)) + j * w, [100 * s["pixel_share_of_all_pixels"][c] for c in classes], w, label=k)
    axes[0].set_yscale("log"); axes[0].set_title("Annotated instances per class (log)"); axes[1].set_title("Pixel share of all pixels (%)")
    for a in axes:
        a.set_xticks(np.arange(len(classes)) + 0.4); a.set_xticklabels(classes, rotation=20)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "class_frequencies.png", dpi=90); plt.close(fig)

    tr = [k for k in keys if k.endswith("/train")]
    fig, axes = plt.subplots(1, len(tr), figsize=(6 * len(tr), 5), squeeze=False)
    for ax, k in zip(axes[0], tr):
        m = np.array(summary["splits"][k]["pixel_cooccurrence_P_b_given_a"])
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30); ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for a in range(len(classes)):
            for b in range(len(classes)):
                ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", color="w" if m[a, b] < 0.6 else "k", fontsize=8)
        ax.set_title(f"P(col | row) pixel co-occurrence\n{k}")
    fig.tight_layout(); fig.savefig(out / "cooccurrence.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, k in enumerate(tr):
        s = summary["splits"][k]["instance_char_size_at_eval_res_px"]
        for i, c in enumerate(classes):
            ax.errorbar(i + 0.1 * j, s[c]["p50"], yerr=[[s[c]["p50"] - s[c]["p5"]], [s[c]["p95"] - s[c]["p50"]]], fmt="o", capsize=3, label=k if i == 0 else None)
    ax.set_yscale("log"); ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes); ax.set_ylabel("sqrt(area) at 512px eval resolution (median, p5-p95)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "instance_sizes.png", dpi=90); plt.close(fig)

    L = ["# EDA report (auto-generated by scripts/run_eda.py)\n",
         f"Classes: {', '.join(classes)}. dacl10k strategy: `{summary['dacl10k_test_strategy']}`.\n",
         "Figures: `class_frequencies.png`, `cooccurrence.png`, `instance_sizes.png`, `samples_*.png`.\n"]
    for k in keys:
        s = summary["splits"][k]
        L += [f"\n## {k}\n", f"{s['n_images']} images ({s['n_images_with_any_target']} with >=1 target-class annotation); {s['n_distinct_resolutions']} distinct resolution(s), most common: {s['resolutions']}\n",
              "| class | instances | % images containing | % of all pixels | median sqrt(area) @512px | median bbox aspect |", "|---|---|---|---|---|---|"]
        for c in classes:
            L.append(f"| {c} | {s['instances'][c]} | {100 * s['image_fraction_with_class'][c]:.1f} | {100 * s['pixel_share_of_all_pixels'][c]:.2f} | {s['instance_char_size_at_eval_res_px'][c]['p50']:.0f} | {s['instance_bbox_aspect'][c]['p50']:.1f} |")
    for name, st in summary.get("image_stats_train", {}).items():
        L.append(f"\n## Image statistics: {name}/train (n={st['n_sampled']})\n\nRGB mean {st['rgb_mean']}, mean within-image std {st['rgb_std_within_image']}, brightness p5/p50/p95 {st['brightness_mean_p5_p50_p95']}, images with >5% near-black pixels: {100 * st['frac_images_with_gt5pct_near_black_pixels']:.1f}%.")
    (out / "eda_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
