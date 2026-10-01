"""Albumentations pipelines. Geometric + photometric for train; resize/normalize only for val/test.

A prior sibling project's augmentation was photometric-only (blur + color jitter) — this project
adds geometric transforms (flip/rotate90/shift-scale-rotate) as new coverage, since damage on a
structure surface has no canonical up/down orientation and geometric augmentation increases
effective training diversity more than photometric alone. Validation/test data is never
augmented — augmenting the data you use to judge overfitting adds noise to exactly the signal
you're trying to read (see CLAUDE.md).
"""
from __future__ import annotations

import albumentations as A

from configs.schema import AugmentationConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(aug_cfg: AugmentationConfig, image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=aug_cfg.horizontal_flip_p),
            A.VerticalFlip(p=aug_cfg.vertical_flip_p),
            A.RandomRotate90(p=aug_cfg.rotate90_p),
            A.Affine(
                translate_percent=(-aug_cfg.shift_limit, aug_cfg.shift_limit),
                scale=(1.0 - aug_cfg.scale_limit, 1.0 + aug_cfg.scale_limit),
                rotate=(-aug_cfg.rotate_limit, aug_cfg.rotate_limit),
                border_mode=0,  # constant padding — don't reflect defect masks into new geometry
                p=aug_cfg.shift_scale_rotate_p,
            ),
            A.GaussianBlur(
                sigma_limit=(aug_cfg.blur_sigma_min, aug_cfg.blur_sigma_max),
                p=aug_cfg.blur_p,
            ),
            A.ColorJitter(
                brightness=aug_cfg.brightness_limit,
                contrast=aug_cfg.contrast_limit,
                saturation=0.2,
                hue=aug_cfg.hue_shift_limit / 360.0,  # albumentations ColorJitter hue is a fraction of the hue circle
                p=aug_cfg.color_jitter_p,
            ),
            A.RandomShadow(
                shadow_intensity_range=(aug_cfg.shadow_intensity_min, aug_cfg.shadow_intensity_max),
                p=aug_cfg.shadow_p,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
