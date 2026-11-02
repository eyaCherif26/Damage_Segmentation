#!/usr/bin/env bash
# Batch version of scripts/infer_orthophoto.py: runs it once per image found in IMAGE_DIR, each
# into its own subfolder of OUTPUT_ROOT. A per-image subfolder is required, not optional --
# infer_orthophoto.py writes fixed filenames per run (prob_<class>.png, overlay.png,
# predictions.coco.json, run_info.json), so two images sharing one --output_dir would silently
# overwrite each other's outputs.
#
# Runs images SEQUENTIALLY, one at a time, not in parallel/backgrounded: this project's GPU budget
# is one 8GB card (see CLAUDE.md), and running several infer_orthophoto.py processes at once would
# contend for it rather than speed anything up.
#
# A single failing image (corrupt file, OOM, etc.) does not abort the whole batch -- its path is
# recorded in OUTPUT_ROOT/failed.txt and the loop continues to the next image.
#
# Optional post-processing (opt-in, choose per run): anything after a literal `--` is forwarded
# verbatim to scripts/postprocess_predictions.py, run against each image's own --output_dir right
# after infer_orthophoto.py succeeds for it (writing to <out_dir>/postprocessed/). Omit the `--` and
# everything after it entirely to skip post-processing -- the default, unchanged behavior. Pick
# whichever combination of --hysteresis_low_threshold/--morph_classes/--morph_close_px/
# --morph_open_px/--merge_classes/--merge_gap_px/--merge_spur_prune_px you want per run; none of
# them are hardcoded here (see scripts/postprocess_predictions.py --help for all of them).
#
# Usage:
#   scripts/infer_orthophoto_batch.sh IMAGE_DIR OUTPUT_ROOT CONFIG CHECKPOINT THRESHOLDS [-- POSTPROCESS_ARGS...]
#
# Example (no post-processing, original behavior):
#   scripts/infer_orthophoto_batch.sh \
#     /home/eyacherif/Projects/new_data_infracloud/autobahn_sel/selections/clips/Schleusenturm \
#     viz/segformer_native_cif_512/autobahn_sel \
#     configs/experiments/segformer_native_cif_512.yaml \
#     runs/segformer_native_cif_512/best.pt \
#     results/segformer_native_cif_512/thresholds.json
#
# Example (with post-processing):
#   scripts/infer_orthophoto_batch.sh \
#     /home/eyacherif/Projects/new_data_infracloud/hafen_v11_annotationen/bildern \
#     viz/segformer_native_cif_512/infra_hafen/ki_results \
#     configs/experiments/segformer_native_cif_512.yaml \
#     runs/segformer_native_cif_512/best.pt \
#     results/segformer_native_cif_512/thresholds.json \
#     -- --hysteresis_low_threshold Cracks=0.1 Spalling=0.1 Efflorescence=0.1 \
#        --morph_classes Cracks --morph_close_px 2 --morph_open_px 1 \
#        --merge_classes Cracks --merge_gap_px 30
set -uo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 IMAGE_DIR OUTPUT_ROOT CONFIG CHECKPOINT THRESHOLDS [-- POSTPROCESS_ARGS...]" >&2
    exit 1
fi

IMAGE_DIR=$1
OUTPUT_ROOT=$2
CONFIG=$3
CHECKPOINT=$4
THRESHOLDS=$5
shift 5

POSTPROCESS_ARGS=()
if [ "$#" -gt 0 ]; then
    if [ "$1" != "--" ]; then
        echo "Usage: $0 IMAGE_DIR OUTPUT_ROOT CONFIG CHECKPOINT THRESHOLDS [-- POSTPROCESS_ARGS...]" >&2
        echo "(extra arguments must come after a literal '--', e.g. -- --hysteresis_low_threshold Cracks=0.1)" >&2
        exit 1
    fi
    shift
    POSTPROCESS_ARGS=("$@")
fi

mkdir -p "$OUTPUT_ROOT"
IMAGE_LIST="$OUTPUT_ROOT/image_list.txt"
FAILED_LOG="$OUTPUT_ROOT/failed.txt"
POSTPROCESS_FAILED_LOG="$OUTPUT_ROOT/postprocess_failed.txt"
: > "$POSTPROCESS_FAILED_LOG"

# One path per line, case-insensitive on common raster extensions; --maxdepth 1 (not recursive) to
# match exactly "every image in this folder", not sub-folders too.
find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" \) | sort > "$IMAGE_LIST"
total=$(wc -l < "$IMAGE_LIST")
echo "[batch] found $total images in $IMAGE_DIR -> $IMAGE_LIST"
: > "$FAILED_LOG"

i=0
while IFS= read -r image; do
    i=$((i + 1))
    stem=$(basename "$image")
    stem="${stem%.*}"
    out_dir="$OUTPUT_ROOT/$stem"
    echo "[batch] ($i/$total) $stem -> $out_dir"
    if ! python scripts/infer_orthophoto.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --thresholds "$THRESHOLDS" \
        --image "$image" --output_dir "$out_dir" --segmentation_format polygon; then
        echo "[batch] ($i/$total) FAILED: $stem"
        echo "$image" >> "$FAILED_LOG"
        continue
    fi
    if [ "${#POSTPROCESS_ARGS[@]}" -gt 0 ]; then
        echo "[batch] ($i/$total) postprocessing $stem -> $out_dir/postprocessed"
        if ! python scripts/postprocess_predictions.py --input_dir "$out_dir" "${POSTPROCESS_ARGS[@]}"; then
            echo "[batch] ($i/$total) postprocessing FAILED: $stem"
            echo "$image" >> "$POSTPROCESS_FAILED_LOG"
        fi
    fi
done < "$IMAGE_LIST"

n_failed=$(wc -l < "$FAILED_LOG")
echo "[batch] done: $((total - n_failed))/$total succeeded -> $OUTPUT_ROOT/<image_name>/"
if [ "$n_failed" -gt 0 ]; then
    echo "[batch] $n_failed failed, see $FAILED_LOG"
fi
if [ "${#POSTPROCESS_ARGS[@]}" -gt 0 ]; then
    n_pp_failed=$(wc -l < "$POSTPROCESS_FAILED_LOG")
    if [ "$n_pp_failed" -gt 0 ]; then
        echo "[batch] $n_pp_failed postprocessing failures, see $POSTPROCESS_FAILED_LOG"
    fi
fi
