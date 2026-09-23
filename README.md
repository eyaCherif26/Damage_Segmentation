# MVP_clean — multi-label infrastructure damage segmentation

Multi-label semantic segmentation of **Cracks, Wetspot, Spalling, Rust, Efflorescence** (one sigmoid channel per class, masks may overlap) from two COCO-format datasets
(cif_coco + dacl10k), with three swappable baseline models — **SegFormer (MiT-b0)**, **smp + EfficientNet-B0**, **smp + MobileNetV3-Large** — each with ImageNet-only or dacl/bikit "task-pretrained" initialisation,
W&B tracking, per-class threshold selection, and sliding-window inference for large orthophotos. Everything runs on an 8 GB GPU (default configs peak at 2.2–3.2 GB).

Read next: [CLAUDE.md](CLAUDE.md) (decisions and pitfalls) · [docs/dataset_report.md](docs/dataset_report.md) (datasets, label mapping, domain gap) · [docs/methodology.md](docs/methodology.md) (design rationale, evaluation protocol) · [docs/eda/eda_report.md](docs/eda/eda_report.md).

```
configs/      schema.py + loader.py (dataclass+YAML), data/{datasets,label_maps}, experiments/*.yaml
data/         merge_dacl10k.py, common_schema.py (COCO->masks), label_mapping.py, validity_mask.py, split_utils.py, datasets/ (incl. crop_dataset.py, image_folder_dataset.py)
models/       registry.py, smp_model.py, segformer_model.py, torchvision_encoders.py, bikit_adapter.py, segformer_pretrained.py, init_modes.py
losses/       focal.py, dice_tversky.py, boundary.py (soft clDice), combined.py
metrics/      pixel_metrics.py, boundary_metrics.py, instance_metrics.py, evaluation.py
training/     trainer.py, schedule.py, early_stopping.py, wandb_logger.py
inference/    sliding_window.py, vram_budget.py, postprocess.py, visualize.py
scripts/      validate_datasets.py, run_eda.py, train.py, evaluate.py, visualize_predictions.py, infer_orthophoto.py   (the dacl10k merge CLI is data/merge_dacl10k.py)
tests/        221 tests (pytest)
```

## 1. Set up the environment

Requirements: Linux/WSL2, an NVIDIA GPU with ≥ 8 GB, Python 3.12, conda.

```bash
conda create -n mvp_clean_env python=3.12 -y
conda activate mvp_clean_env
# 1) PyTorch first, with the CUDA build matching your driver (tested: 2.8.0 + cu128)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# 2) everything else (exact tested versions are in requirements-freeze.txt)
pip install -r requirements.txt
# 3) check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests -q          # expect: 221 passed
```

Notes: the first SegFormer / smp run downloads ImageNet weights (Hugging Face hub and/or GitHub releases) (a `HF_TOKEN` is optional, it only lifts rate limits).
Run all commands from the repository root. Don't cap memory with `ulimit -v` — it breaks cuDNN loading (looks like a model bug, isn't).

## 2. Point the configs at your data

Nothing in the code hard-codes a dataset location. Paths live in `configs/data/datasets/*.yaml` — one file per source with, **per split**, an `images_roots` directory and an `annotations` COCO JSON, plus a `label_mapping_path`:

```yaml
name: cif_coco
images_roots: {train: /data/cif/train/images, val: /data/cif/val/images, test: /data/cif/test/images}
annotations:   {train: /data/cif/train/annotations.json, val: ..., test: ...}
label_mapping_path: configs/data/label_maps/cif6_to_mvp5.yaml
native_categories: [Algae, Crack, NetCrack, CrackPrecip, Rust, Spalling]
```

Edit `cif_coco.yaml` and `dacl10k.yaml` to your paths (relative paths resolve against the YAML's own directory; `label_mapping_path` against the repo root).
Both must be **COCO with polygon or RLE segmentation and 1-based category ids**.

## 3. Merge dacl10k (per-image labelme JSON → one COCO file per split)

dacl10k ships one labelme-style JSON per image. Merge them (skip this if you already have per-split COCO files; the script is generic and takes any labelme-style directory pair):

```bash
python data/merge_dacl10k.py \
  --train_ann_dir <dacl10k>/annotations/train      --train_img_dir <dacl10k>/images/train \
  --val_ann_dir   <dacl10k>/annotations/validation --val_img_dir   <dacl10k>/images/val \
  --output_dir data/processed/dacl10k_merged
```

Output: `data/processed/dacl10k_merged/{train,val}/annotations.json` (+ `merge_report.json` with per-class counts, zero-annotation images, dropped degenerate polygons). Category ids are assigned 1..N over the sorted union of labels (pass `--categories_file` to pin an order); polygons only — any other `shape_type` raises. Refuses to overwrite without `--force`. dacl10k's `testdev` has no labels and is not merged.
Then set `annotations:` in `configs/data/datasets/dacl10k.yaml` to these two files.

## 4. Validate the data before anything else

```bash
python scripts/validate_datasets.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml
```

Checks every split of every source: required COCO keys, 1-based ids, null/empty segmentation, box-only polygons, dangling image ids, sampled image files exist, and that **every native category has an entry in the label map**. Exits non-zero on failure.
It also creates the persisted dacl10k val carve on first use (see below).

**dacl10k has no labeled test set.** Default (`data.dacl10k_test_strategy: carve_val_from_train`): a fixed-seed 10% of dacl10k *train* becomes val, and the official *validation* split becomes *test*. The carve is written once to `data/splits/dacl10k_train_val_carve_seed42.json` and is **read, never recomputed** — commit it to keep experiments comparable. Option (b), `val_only`, uses the official val as both val and test.
The cif_coco splits are used exactly as provided. Default training is on the **combined** data (`data.sources: [cif_coco, dacl10k]`); use `[cif_coco]` or `[dacl10k]` for single-source runs.

## 5. EDA

```bash
python scripts/run_eda.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml --output_dir docs/eda   # ~30 s
```

Class frequencies, instance sizes (px at native and at 512 eval resolution, plus image-area fractions; mm is not reported because source GSD is unknown), overlap co-occurrence matrix, image statistics, sample overlays, all under the unified 5-class schema and the exact train/val/test roles used for training. `--skip_images` skips everything that reads image files.

## 6. Train

Experiment configs in `configs/experiments/`: `baseline_{smp_efficientnet_b0, smp_mobilenet, segformer}.yaml` (ImageNet init), `baseline_smp_efficientnet_b0_task_pretrained.yaml`, `segformer_task_pretrained.yaml`, `mobilenet_task_pretrained.yaml` (task-pretrained init, cif + dacl10k), `effb0_task_pretrained_cif.yaml` and `segformer_task_pretrained_cif.yaml` (cif_coco only, like-for-like pair), `effb0_task_pretrained_dacl.yaml`, `segformer_task_pretrained_dacl.yaml` and `segformer_imagenet_dacl.yaml` (dacl10k only; the last is the leakage-free SegFormer comparison).

**Smoke test first** (≈ 1 min; a seeded random subset, W&B offline so no credentials needed):

```bash
python scripts/train.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml \
  --wandb_mode offline --wandb_entity smoke --wandb_project smoke \
  --max_epochs 3 --max_train_images 320 --max_val_images 96 --checkpoint_dir /tmp/mvp_smoke
```

**Full run.** W&B: `wandb login` once; the experiment YAMLs are set to `entity: eya-cherif14`, `project: damage_detection` (the `WandbConfig` default is `SET_ME`, which makes training refuse to start — override with `--wandb_entity/--wandb_project`, or use `--wandb_mode disabled`):

```bash
python scripts/train.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml \
  --wandb_entity <your-entity> --wandb_project <your-project>
```

Measured speed on an RTX 3050 8 GB (24-core CPU, 8 workers): ~27 img/s, i.e. **≈ 9.5 min per epoch** on the combined data (13,267 train / 2,311 val images); it is data-loading bound (large JPEG decode + native-resolution rasterisation), so all three models take the same time. Early stopping (patience 5 on val macro Dice) can end a run before the 30-epoch ceiling (a full run has not been done yet, so the actual stopping epoch is unknown); worst case ≈ 5 h.
Outputs in `runs/<experiment name>/`: `best.pt` (best val macro Dice), `last.pt`, `resolved_config.yaml`. There is no resume-from-checkpoint. Per epoch, W&B gets train/val loss, train/val macro Dice/IoU, `gap/*` (train–val), LRs, a per-class metric table, and (every 5 epochs) prediction-vs-GT overlays.

**Initialisation** (`model.init_mode`): `from_scratch` = ImageNet encoder only (the default). `task_pretrained` loads weights from a model trained on a similar task:
- **SegFormer** — `segformer_task_pretrained.yaml` loads a **dacl10k-trained MiT-b1** (encoder + decode head; the classifier is fresh because that checkpoint has 19 classes). `encoder_name` must match the checkpoint's MiT variant, else it raises. Set `model.load_classifier: true` only for a checkpoint with the same class count *and order*.
- **EfficientNet-B0** — `baseline_smp_efficientnet_b0_task_pretrained.yaml` loads the bikit/dacl encoder (99.2% tensor coverage).
- **MobileNetV3-Large** — `mobilenet_task_pretrained.yaml` loads a dacl1k bikit encoder (308/308 tensors). This works because the encoder is torchvision's, matching the checkpoints' format. Details in CLAUDE.md.
Verify a task-pretrained run really used the dacl encoder: the log prints `[init] task_pretrained load report: ... coverage 0.99...`, and `python scripts/check_pretrained_init.py --config <yaml> [--checkpoint runs/<name>/best.pt]` compares weights bit-exactly (CPU, safe during training).
LR sweep like the dacl10k paper: copy the YAML and change `training.lr_head` / `lr_backbone` (1e-4 … 5e-3); not automated.

### Training with less resizing (native resolution)

By default every image is squeezed to 512x512 (the dacl10k-paper recipe). Photos are often far larger (dacl10k: median ~x0.28 shrink, up to 24 MP), so fine detail is lost. Set `data.resize_mode: native_crop` to train on native pixels instead — ready-made configs (ImageNet-pretrained encoder, `init_mode: from_scratch`): `effb0_native.yaml`, `segformer_native.yaml`, `mobilenet_native.yaml`.

```bash
python scripts/train.py --config configs/experiments/effb0_native.yaml
```

```yaml
data:
  resize_mode: native_crop     # whole_image (default) | native_crop
  pixel_scale: 1.0             # 1.0 = native pixels; 0.5 = half resolution (more context, less detail)
  train_scale_jitter: 0.25     # each training crop's scale is drawn log-uniformly in [s/1.25, s*1.25]
  fg_crop_prob: 0.7            # share of crops centred on a labelled object (the rest: uniform random)
  train_crops_per_image: 1     # crops per image per epoch (epoch length scales with it)
  val_crops_per_image: 1       # fixed crops per val image for per-epoch monitoring
  image_size: 512              # = crop / tile size
```

How it works: the image is scaled by `pixel_scale`, and the network sees 512 px windows. **Train** = random windows (with `fg_crop_prob` centred on a random point of a labelled object, so crops are not mostly empty). **Val (monitoring)** = the same deterministic windows every epoch. **Test / `evaluate.py`** = a full grid of non-overlapping tiles covering every pixel of every image (thresholds are also chosen on a full val grid, which is unbiased, unlike the object-centred monitoring crops). Labels are rasterised straight into each window (no 24 MP masks), image and labels share one geometric mapping (tested for alignment at several scales), windows beyond the image edge are padded and excluded from loss/metrics, and images are decoded with OpenCV's default EXIF-aware decoder and size-checked against the annotations.
Trade-offs: (1) each sample sees **less scene context** (a 512 px window is 1/8 of a 4000 px photo at scale 1.0) — very large regions like Wetspot may not fit in one window; lower `pixel_scale` (0.5, 0.35) for a compromise. (2) "Native" is not a fixed physical scale: photos are taken from different distances, so pixel size in mm varies per image (`train_scale_jitter` helps robustness). (3) `evaluate.py` becomes much slower (about 19k val + 21k test tiles at scale 1.0 instead of ~4.4k images). (4) A model trained at `pixel_scale` s expects objects to look like they do at that scale — pick `infer_orthophoto.py --scale` accordingly (section 9), or train directly on your own orthophotos at native 2 mm/px and use `--scale 1.0`.
Measured: the crop loader delivers ~104 samples/s vs ~11 for whole-image loading in the same (CPU-loaded) conditions, because only the window is decoded/rasterised into memory; the network step is unchanged (same 512x512 input, same VRAM). `visualize_predictions.py` shows tiles at `pixel_scale` for such models, and refuses `--image_dir` (which would resize whole images) — use `infer_orthophoto.py` there.

## 7. Evaluate

```bash
python scripts/evaluate.py --config configs/experiments/baseline_smp_efficientnet_b0.yaml \
  --checkpoint runs/baseline_smp_efficientnet_b0/best.pt --output_dir results/baseline_smp_efficientnet_b0
```

Order of operations: (1) GT-fed-back-as-prediction sanity check (aborts if the eval code disagrees with itself), (2) **per-class thresholds chosen on val only**, (3) those thresholds applied to val and test. Writes `results_test.csv`, `results_val.csv`, `thresholds.json`.
CSV columns: `split, scope (overall | source:<name>), class (per class | macro | frequency_weighted), threshold, iou, dice, precision, recall, gt_pixels, boundary_f1, cldice (thin classes only), inst_recall_{small,medium,large}, inst_n_gt_*`. All metrics are computed inside the validity mask. `--max_images N` for a quick check.
Interpret pooled numbers with care: the two sources have different label semantics (see the dataset report) — read the `source:*` rows.

### Evaluate on another labelled dataset

Any dataset with a COCO annotation file and an image folder (paths are plain arguments, nothing is assumed about its layout):

```bash
python scripts/evaluate.py --config configs/experiments/segformer_task_pretrained_cif.yaml \
  --checkpoint runs/segformer_task_pretrained_cif/best.pt --output_dir results/segformer_task_pretrained_cif \
  --coco path/to/annotations.json --images path/to/images [--label_map path/to/map.yaml] [--name my_data]
```

- **Labels.** Without `--label_map` the file's categories must already be MVP class names (`Cracks, Wetspot, Spalling, Rust, Efflorescence`); any other name aborts. With a label map every category needs an entry (`null` = counted as background), e.g. `configs/data/label_maps/dacl10k_19_to_mvp5.yaml` for raw dacl10k.
- **Thresholds are never tuned on the new data.** They come from `--thresholds FILE`, else `<output_dir>/thresholds.json` (written by a normal `evaluate.py` run), else they are selected on the *configured* val split first and saved. So the numbers measure how the model transfers, not how well it was fitted to that dataset.
- **Output** `results_<name>.csv` (same columns as above; `split` = `external:<name>`; `--name` defaults to the annotation file's folder name). `results_val/test.csv` and `thresholds.json` of the original run are left untouched.
- **Classes the dataset has no labels for** (e.g. a dataset that only labels cracks) show `gt_pixels 0` and are announced at start-up. `macro` still averages all five classes, so use the extra **`macro_present`** row (mean over classes that occur in the ground truth) for such data.
- `native_crop` models are evaluated on a full tile grid at their `pixel_scale`, exactly like the configured test split. The GT-as-prediction sanity check runs on the new data first, so a broken label map or misaligned annotations fail before any number is reported. `--max_images N` for a quick look.

## 8. Look at predictions (qualitative review)

Choose **any** data — not only the test split — and images are drawn **at random, freshly on every run** (the seed is printed and saved in `selection.json`; add `--seed N` to reproduce a draw).

```bash
# (a) configured split (default: test): 12 random images
python scripts/visualize_predictions.py \
  --model configs/experiments/effb0_task_pretrained_cif.yaml runs/effb0_task_pretrained_cif/best.pt \
  --split val --num_images 12 --output_dir viz/effb0_cif

# (b) NEW LABELLED DATA: any COCO file + its image folder (paths are just arguments)
python scripts/visualize_predictions.py --model <config> <checkpoint> \
  --coco path/to/annotations.json --images path/to/images --output_dir viz/new_labelled
#     add  --label_map my_map.yaml  if its categories are not already the 5 class names
#     (unmapped categories raise instead of being dropped; map ones to ignore to `null`)

# (c) NEW UNLABELLED IMAGES: any folder (add --recursive for sub-folders); prediction-only panels
python scripts/visualize_predictions.py --model <cfgA> <ckptA> --model <cfgB> <ckptB> \
  --image_dir path/to/new/images --num_images 16 --output_dir viz/new_unlabelled
```

Panels: image | ground truth (only if labels exist) | one column per model, so several models are compared on identical images. Classes are translucent fills plus a contour in their own colour (overlaps stay readable); tinted, cyan-outlined regions are no-data pixels excluded from loss and metrics. With labels each model column shows macro Dice (over the classes present in the ground truth) and per-class Dice; without labels it lists the predicted classes.
`--model CONFIG CHECKPOINT [THRESHOLDS_JSON]`: thresholds default to `results/<experiment name>/thresholds.json` from `scripts/evaluate.py` (val-selected) and fall back to 0.5 with a warning.
Only where ground truth exists: `--select worst|best` (ranked by the first model), `--source`, `--require_class` — on unlabelled images these fail with a clear message.
Images are shown at the 512 px evaluation resolution the model sees (whole image resized, as in training); for large orthophotos use section 9. Folders may contain Windows `*:Zone.Identifier` files — only real image extensions are read.

## 9. Inference on a large orthophoto (tiles)

```bash
python scripts/infer_orthophoto.py \
  --config configs/experiments/effb0_task_pretrained_cif.yaml --checkpoint runs/effb0_task_pretrained_cif/best.pt \
  --image path/to/orthophoto.png --output_dir out/ortho --scale 0.5
```

What happens: the image is (optionally) rescaled by `--scale`, cut into overlapping 512 px tiles (`--overlap 128`, the last row/column flush with the edge), every tile goes through the model, and the tile probabilities are **averaged where tiles overlap** (no seams), black no-data borders are zeroed, per-class thresholds from `results/<experiment>/thresholds.json` are applied (val-selected by `scripts/evaluate.py`; 0.5 with a warning if absent), and everything is mapped back to the original pixel grid.
Outputs in `--output_dir`: `overlay.png` (+ `overlay_preview.jpg` when the image is larger than `--preview_max_side`, default 4096, because a giant PNG is slow to open), `prob_<class>.png` (8-bit probability per class), `predictions.coco.json` (RLE, one instance per connected component, scored by mean probability, components below `--min_area` dropped) and `run_info.json` (all settings + overlay colours, so runs can be compared). Overlay colours match `visualize_predictions.py`.

**Choose `--scale` first — it matters most and it is not known a priori.** Training images were downscaled to 512 px (cif_coco x0.5; dacl10k typically x0.15-0.5) and neither dataset records its GSD, so the right factor for a 2 mm/px orthophoto has to be found empirically. Try a few on a small window of a damaged area before processing the whole image (`--crop X Y W H`, in original pixels):

```bash
for s in 0.25 0.35 0.5 0.75 1.0; do
  python scripts/infer_orthophoto.py --config <cfg> --checkpoint <ckpt> --image ortho.png \
    --crop 4000 3000 2048 2048 --scale $s --output_dir out/scale_$s
done      # then compare out/scale_*/overlay.png
```

What `--scale` does: the image is resized by that factor *before* tiling, so a 512 px tile covers `512/scale` original pixels and objects appear `scale` times as large; the result is resampled back to the original grid (output resolution never changes). It should make objects look about as big (in pixels) as they did in training. Measured example (EfficientNet-B0 cif model, 80 real 1024 px cif test images, whose training scale is exactly x0.5): macro Dice 0.40 at x0.25, 0.54 at x0.35, **0.57 at x0.5**, **0.57 at x0.7**, 0.50 at x1.0; at x0.25 Wetspot collapses (Dice 0.06) and large spurious regions appear, at x1.0 Efflorescence and Cracks drop while cost rises from 1 to 9 tiles per image. Scale and tile context are confounded at >= x0.7 (each tile then sees less of the scene), and the optimum depends on what scale the training data had, so this shows the shape of the effect, not the right value for your orthophoto.

With `--crop`, outputs and COCO coordinates are relative to the crop (its origin is in `run_info.json`). Calibrate on labelled orthophoto tiles as soon as you have some (`visualize_predictions.py --coco ... --images ...` shows them at 512 px only; run this script on the same crop for the tiled result).
Limits: measured on a 120 MP image (0.5 scale, CPU): ~6 min and 7.3 GB peak RAM; memory grows linearly with pixel count (the full-resolution probability maps are held in RAM), so roughly 400+ MP will not fit in ~32 GB — use `--crop` windows or a coarser `--scale`. `--max_memory_gb` (default 8) refuses to allocate the tile accumulator beyond that. The output is plain PNG/JSON in pixel coordinates: **georeferencing (GeoTIFF/CRS) is not preserved**.

## 10. Extending

- **New dataset**: add `configs/data/datasets/<name>.yaml` (paths per split) and a label map `configs/data/label_maps/<name>_to_mvp5.yaml` covering *every* native category (unlisted categories fail loudly; map non-target ones to `null`); add the name to `data.sources` and `data.dataset_config_paths`. No code changes.
- **New damage class / material**: append to `data.mvp_classes`, set `model.num_classes`, edit the label maps (`configs/schema.py::ExperimentConfig` cross-checks these and any per-class loss keys).
- **New backbone**: write a factory `f(ModelConfig) -> nn.Module` returning full-resolution logits `(B, C, H, W)` with `backbone_parameters()` / `head_parameters()`, decorate with `@register_model("name")`, import it in `models/__init__.py`. Reference it from an experiment YAML.
- **New label semantics** (e.g. split `Weathering` out of `Wetspot`): edit the label-map YAML + `mvp_classes`; `tests/test_label_mapping.py` pins today's mapping so the change is deliberate.

## 11. Tests

```bash
python -m pytest tests -q
```

221 tests: COCO→mask conversion (polygon/RLE, multi-label overlap), label-map coverage, dacl10k merge (fixtures), split determinism/persistence, focal/Tversky/clDice losses (validity-mask leak tests), pixel/boundary/instance metrics incl. GT-as-prediction, threshold sweep, scheduler/freezing invariants, model contract for every registered model and decoder, bikit adapter guard, SegFormer legacy-checkpoint loading (bit-exact round-trip, pickled-object recovery, architecture guard), torchvision MobileNet encoder (stage-output equality, 308/308 checkpoint mapping, dropout actually applied), preprocessing consistency (same transform for every model, matches each backbone's expected normalisation, masks stay binary, train==inference), prediction visualisation (class overlays, no-data marking, OpenCV memory-layout regression, selection filters), validity-mask detection, sliding-window tiling/blending/memory guard, config validation.

## Known limitations (see docs for detail)

Not implemented: multi-scale TTA, class-aware sampling, copy-paste, child-level (hierarchical) outputs, mask AP, resume-from-checkpoint, degradation augmentation beyond blur/shadow/colour. Domain gap to harbor orthophotos (materials, viewpoint, GSD, tidal/underwater effects) is documented, not solved — only labelled orthophoto tiles can measure it.
Licences: dacl10k is CC BY-NC 4.0 (non-commercial); cif_coco is CDLA-Permissive-2.0 (per its Hugging Face card).
