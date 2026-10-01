# Dataset report (Phase 0)

All numbers below were measured on the local copies (not recalled), by `data/merge_dacl10k.py`, `scripts/validate_datasets.py`
and `scripts/run_eda.py` (full tables/figures in [eda/eda_report.md](eda/eda_report.md)). Items I could not verify are marked `[unverified]`.

## 1. Sources at a glance

| | cif_coco | dacl10k (v2) |
|---|---|---|
| What | IBM Research "Cracks in the Foundation" (2025), civil-infrastructure inspection tiles | Flotzinger, Rösch, Braml — bridge-inspection photos, *WACV 2024* / dacl-challenge |
| Local format | COCO, polygon segmentation, one JSON per split | one labelme JSON **per image** (polygon only) -> merged by `data/merge_dacl10k.py` into one COCO file per split |
| Splits provided | train / val / test | train / validation / testdev (**testdev has images but no labels**) |
| Images used | 7,026 / 1,617 / 1,115 (train/val/test) | 6,935 / 975 (train/val), 1,012 unlabeled testdev unused |
| Resolution | all 1024×1024 | variable: 311 distinct sizes in train (most common 1280×960, 1024×768, 1920×1440, 4000×3000; up to 6000×4000) |
| GSD | not documented `[unverified]` | not documented (handheld/smartphone photos, taken 2000–2020, Germany) |
| Native classes | 6: Algae, Crack, NetCrack, CrackPrecip, Rust, Spalling | 19 (13 damage + 6 object, see §2) |
| Annotation count used | 20,809 / 4,888 / 3,231 | 54,158 / 8,167 |
| License | **CDLA-Permissive-2.0** (per the Hugging Face dataset card, fetched 2026-09-21; no license file exists locally) | **CC BY-NC 4.0** (`LICENSE.md` shipped with the data) — non-commercial only; bounds any sharing of checkpoints trained on it |

Cross-checks that passed: the merge reproduces the per-class totals of an independent full scan (e.g. Rust 12,837 + 1,883 = 14,720; Cavity 8,119 + 1,152 = 9,271),
0 degenerate polygons, `validate_datasets.py` passes for every split (1-based category ids, no null segmentation, no box-only polygons, all label-map entries present).

### cif_coco is a class-balanced *subset* — verified
The folders hold 105,139 / 22,368 / 21,135 tiles (matching the HF card's tiled counts), but the working `annotations.json` covers only **7,026 / 1,617 / 1,115**
(`annotations_balanced_subset.json` is byte-identical to it; the full file `annotations_all_ori.json` still exists). Original -> subset instance counts (train):
Crack 90,823 -> 5,003; Net-Crack 10,881 -> 3,113; Crack+Precip 3,057 -> 3,057 (all kept); Rust 11,469 -> 3,136; Spalling 28,442 -> 3,281; Algae 32,997 -> 3,219.
Consequences: the cif portion has (a) **artificially balanced class priors** (natural prior is Crack-dominated), (b) **no background-only tiles** (every tile has >=1 target annotation, vs 10% in dacl10k),
so it gives little pure-negative supervision on clean surfaces. Using the full 105k tiles is possible (~15x more data) but was not done: the task says to use cif_coco as provided.

### dacl10k has no labeled test set
Confirmed on disk: `annotations/` has only `train` and `validation`; `images/testdev` (1,012 images) has no annotation files. Following the confirmed decision, a **fixed-seed 10% val subset is carved out of train**
(694 images, persisted in `data/splits/dacl10k_train_val_carve_seed42.json`, never regenerated) and the **official validation split becomes the test set** (975 images); config `data.dacl10k_test_strategy: val_only` selects option (b) instead.
Resulting roles (cif_coco + dacl10k): train 7,026 + 6,241, val 1,617 + 694, test 1,115 + 975 images.
(`dacl_resized/test` is a 32-image hand-curated subset of testdev made by the team — project-internal, not the official benchmark; not used.)

## 2. Class taxonomy, instance counts, and the unified mapping

Target classes (naming taken from the user's cleaned data): **Cracks, Wetspot, Spalling, Rust, Efflorescence**.
Mapping = `configs/data/label_maps/{dacl10k_19_to_mvp5,cif6_to_mvp5}.yaml` (rationale comments inline). It was **reverse-engineered by exact count-matching against the user's existing remapped files**
(every predicted per-class count matched on dacl10k train + val and cif_coco test) rather than guessed — two entries cannot be guessed from names (`CrackPrecip -> Efflorescence`, `Algae -> Wetspot`).
Reference taxonomy: BAW Merkblatt MSV-A 2023 (section numbers verified against the PDF text; the Merkblatt is organised by construction type, so most mappings are approximate).

| source (train count) | -> target | rationale (BAW MSV-A 2023 section) | fit |
|---|---|---|---|
| dacl Crack (3,154), cif Crack (5,003) | Cracks | 3.3 Risse im Massivbau; 5.3 Risse | clean |
| dacl ACrack (376), cif NetCrack (3,113) | Cracks | 3.1.5 Netzrisse = "flächiges Rissbild" (area-wide crack pattern) | approximate — areal pattern, see §4 |
| dacl Rust (12,837), cif Rust (3,136) | Rust | 5.1 Korrosion; 3.4.1 Bewehrung korrodiert | clean |
| dacl Efflorescence (3,450) | Efflorescence | 3.1.6 Ausblühung/Aussinterung; 3.2.x "Ablagerungen aus ausgefälltem Material" | clean |
| cif CrackPrecip (3,057) | Efflorescence | precipitate at cracks = the Merkblatt's efflorescence definition | **FLAG** name suggests Cracks |
| dacl Wetspot (1,443) | Wetspot | 3.1.4 / 3.2.4 Feucht- und Nassfläche | clean |
| dacl Weathering (4,064) | Wetspot | dacl10k paper: visually near-identical to Wetspot, often overlapping; no Merkblatt "weathering" class for concrete (nearest: 3.1.1 systematic surface loss) | **FLAG** not a clean map; Weathering outnumbers Wetspot ~3:1 |
| cif Algae (3,219) | Wetspot | proxy: algae = persistent moisture; Merkblatt has no algae class | **FLAG** proxy label, noisier |
| dacl Spalling (8,484), cif Spalling (3,281) | Spalling | 3.1.2 Oberfläche geschädigt (Abplatzung) | clean |
| dacl ExposedRebars (1,753) | Spalling | 3.1.2 Bewehrung freigelegt (rebar corrosion is assessed separately in 3.4.1) | **FLAG** loses distinct rebar signal |
| dacl Rockpocket (525), Cavity (8,119) | Spalling | "Kiesnest"/"Lunker" exist but under 3.1.3 Oberfläche offen — construction defects, not spalling (3.1.2) | **FLAG** mixes damage with construction defects |
| dacl Hollowareas (1,363), Restformwork (851) | null | not merged in the user's data (verified by count) | **FLAG** PROMPT.md lists "hollow areas" as a Spalling subtype; the user's data does not merge it. No Merkblatt equivalent for Restformwork |
| dacl Graffiti, Bearing, EJoint, Drainage, PEquipment, JTape, WConccor | null | objects / non-damage / out of scope | — |

Null-mapped polygons are **confirmed negatives** for the 5 classes (not ignore regions).
Two notes on wording vs. PROMPT.md: Wetspot absorbing Weathering and Spalling *excluding* Hollowareas/Restformwork come from the user's existing data, not from the prompt text; flip either in the YAML if you disagree
(`tests/test_label_mapping.py` pins the current mapping, so a change is a deliberate two-line edit).

Final annotated instances in the unified schema (all splits): Cracks 15,349 · Wetspot 10,766 · Spalling 26,317 · Rust 19,082 · Efflorescence 8,216.
Train-role instances: Cracks 11,298 · Wetspot 8,183 · Spalling 20,367 · Rust 14,865 · Efflorescence 6,158 — no class is below ~6k instances, so the MVP has no true few-shot class (imbalance is in *pixels*, not instance counts).

## 3. Mask overlap statistics (multi-label evidence)

Pixel co-occurrence P(column | row), train, measured on 256×256 rasterizations:

dacl10k
| | Cracks | Wetspot | Spalling | Rust | Effl. |
|---|---|---|---|---|---|
| Cracks | 1 | **0.38** | 0.04 | 0.01 | 0.17 |
| Wetspot | 0.05 | 1 | 0.04 | 0.03 | 0.03 |
| Spalling | 0.02 | 0.16 | 1 | 0.07 | 0.01 |
| Rust | 0.01 | 0.17 | 0.12 | 1 | 0.03 |
| Effl. | 0.15 | 0.19 | 0.03 | 0.03 | 1 |

cif_coco: off-diagonals are 0.00–0.13 (largest: P(Spalling | Rust) = 0.13, P(Rust | Spalling) = 0.08). **Overlap is essentially a dacl10k phenomenon**: cif_coco tiles behave almost single-label per region.
This validates the multi-label (sigmoid-per-channel) design and means overlaps like rust-in-spalling and cracks-in-wetspot must not be resolved by argmax anywhere (`tests/test_mask_rasterization.py`, `tests/test_losses.py`).

## 4. Findings from the EDA that affect modelling (verified on the data)

1. **Black no-data padding is common in cif_coco** (measured exactly over every image, train split): **10.3%** of all cif_coco train pixels are near-black padding (val 10.6%, test 10.7%); 53.6% of train images have >1% and 23.8% have >20% flagged (dacl10k: 0.14% of pixels, 2.2% of images >1%). (An earlier 200-image sample suggested 12.1% — sampling noise; the exact figure is quoted here.)
   It is now excluded from loss *and* metrics through the validity mask (`data.no_data_validity: true`, `data/validity_mask.py`); only connected near-black regions of >=64 px count, so isolated dark pixels/shadows stay valid.
2. **"Cracks" mixes two label semantics.** cif_coco Crack/NetCrack polygons often outline whole cracked *regions* (Cracks = 17.7% of all cif pixels; median instance size ~44 px at 512, p95 ~480 px) while dacl10k cracks are thin
   lines (1.8% of pixels; median ~32 px, median bbox aspect 2.8 vs 2.1). The Merkblatt itself treats Netzrisse as an area-wide pattern. Thin-structure metrics (clDice) only make sense for line-type cracks; `scripts/evaluate.py` therefore reports **per-source** metrics.
3. **"Wetspot" mixes moisture, weathering and algae-as-proxy** (see §2 flags); it covers 14.7% of dacl10k pixels (median instance ~146 px at 512) — the largest-area class, unlike Spalling/Rust (median ~16–18 px, i.e. small objects).
4. **Very different scales**: median instance size at 512px is 16–18 px for dacl Spalling/Rust vs 146 px for Wetspot. dacl10k crack labels are *coarse polygons*, not hairlines (paper, Sec. 3.4): measured on 150 random dacl10k train images with cracks, mean width (area / skeleton length) is a median **24.8 px at native resolution and 7.6 px after the 512 resize** (the median photo is shrunk x0.28; range x0.10–x1.14). After the resize 10% of crack components are thinner than 3 px and 27.5% thinner than 5 px, and 1.1% vanish entirely (nearest-neighbour mask resize). So shrinking hurts a minority of thin cracks, not most; but the labels being coarse means native resolution would not recover true hairline width either.
5. **Image statistics**: RGB mean ≈ (0.50, 0.51, 0.51) cif vs (0.53, 0.52, 0.49) dacl — close to ImageNet normalisation; mean within-image std ≈ 0.17–0.19.

## 5. Limitations / domain gap to harbor-waterline orthophotos

- **Structures & materials**: cif_coco = civil-infrastructure inspection tiles; dacl10k = German *bridges*, mostly (reinforced) concrete. The five target materials (Stahlbeton, Beton, Stahl, Mauerwerk, Spundwände) are covered unevenly: concrete/reinforced concrete well, **steel and masonry thinly, sheet pile walls essentially not at all**, so Rust is mostly "rust on/through concrete", not corroded steel sheet piles.
- **Viewpoint**: handheld close-range oblique photos vs. nadir orthophotos rendered from tiled 3D models (~2 mm/px). Perspective, occlusion and lighting differ; dacl10k photos also include sky, vegetation, cars.
- **Resolution / GSD**: source GSD unknown and varies per image; training resizes to 512×512 (×0.5 for cif; for dacl10k the mean image is ≈1581×1950, i.e. ≈×0.26–0.32, and 12 MP photos are ≈×0.13), so a 2 mm/px orthophoto must be rescaled to a matching object scale before inference (`scripts/infer_orthophoto.py --scale`). Which scale is right is an open empirical question `[unverified]` — no source has GSD metadata to calibrate against.
- **Blur / degradation**: orthophoto sharpness varies (blur, tile seams, resampling); handled only by augmentation (blur, colour jitter) plus the validity mask — not by matched real degradation data.
- **Underwater / tidal-zone effects**: wet, algae- and barnacle-covered surfaces, splash zones and water-level lines are absent or rare in both sources. The Wetspot/Algae proxy is the only related signal and is weak. Expect false positives on the wet/dry line and marine growth.
- **Unwanted objects** (boats, ropes, fenders, people) and vegetation: no explicit negative class in the MVP; dacl10k contributes only incidental such content (Graffiti/objects are labeled-as-negative).
- **Label noise**: dacl10k cracks are coarsely annotated (paper, Sec. 3.4) — pixel IoU on cracks is intrinsically low (paper, Table 4: Crack IoU 0.286 on test for the best model). cif tiles are near-exclusive per region, so unlabeled co-occurring damage is likely marked as background (false-negative label noise in the other channels).
- **Licensing**: dacl10k CC BY-NC 4.0 restricts commercial use of derived checkpoints; cif_coco CDLA-Permissive-2.0 (per HF card).

## 6. References
- Flotzinger, J., Rösch, P. J., Braml, T. *dacl10k: Benchmark for Semantic Bridge Damage Segmentation.* WACV 2024. arXiv:2309.00460 (verified: title/authors/abstract at arxiv.org/abs/2309.00460; text read from the CVF open-access PDF).
- Flotzinger, J., Rösch, P. J., Benz, C., Ahmad, M., Cankaya, M., Mayer, H., Rodehorst, V., Oswald, N., Braml, T. *dacl-challenge: Semantic Segmentation during Visual Bridge Inspections.* WACV 2024 Workshops (text read from the CVF open-access PDF).
- IBM Research. *Cracks in the Foundation* (`ibm-research/cif-dataset`), 2025, Hugging Face — https://huggingface.co/datasets/ibm-research/cif-dataset (no author list on the card).
- Bundesanstalt für Wasserbau (BAW). *BAWMerkblatt: Schadensbewertung an Verkehrswasserbauwerken der Inspektionskategorie A (MSV-A)*, Ausgabe 2023 (local PDF `project_test_port_damage_Erik/.../Schadenskatalog/`; section citations checked against its extracted text).
