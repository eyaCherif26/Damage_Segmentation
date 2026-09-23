"""Dataclass schema for MVP_clean's config system.

Plain nested dataclasses + YAML (see CLAUDE.md for why, not Hydra/pydantic). Every script calls
`configs.loader.load_experiment_config(path)` to get a fully-resolved `ExperimentConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

MVP_CLASSES: tuple[str, ...] = ("Cracks", "Wetspot", "Spalling", "Rust", "Efflorescence")
THIN_STRUCTURE_CLASSES: tuple[str, ...] = ("Cracks",)


@dataclass
class DatasetSourceConfig:
    """Resolved from one configs/data/datasets/*.yaml file. Each split gets its own images_root
    since sources like cif_coco/dacl10k store images per-split (train/images, val/images, ...),
    not under one shared root."""

    name: str
    images_roots: dict[str, str]  # split -> images dir, e.g. {"train": "...", "val": "..."}
    annotations: dict[str, str]  # split -> COCO json path, e.g. {"train": "...", "val": "..."}
    label_mapping_path: str
    native_categories: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    sources: list[str] = field(default_factory=lambda: ["cif_coco", "dacl10k"])
    dataset_config_paths: dict[str, str] = field(default_factory=dict)
    mvp_classes: list[str] = field(default_factory=lambda: list(MVP_CLASSES))
    image_size: int = 512
    dacl10k_test_strategy: str = "carve_val_from_train"  # or "val_only"
    dacl10k_val_fraction: float = 0.1
    split_seed: int = 42
    batch_size: int = 8
    num_workers: int = 4
    # Exclude near-black no-data padding (found in ~half of cif_coco images) from loss and metrics
    no_data_validity: bool = True
    # How images reach the network:
    #   whole_image : every image is resized to image_size x image_size (the dacl10k-paper recipe).
    #   native_crop : images are NOT squeezed. `pixel_scale` (1.0 = native pixels) is applied, then
    #                 train = random image_size x image_size crops, val = fixed crops, test = a full
    #                 non-overlapping grid of tiles covering every pixel. Keeps fine detail; trades away
    #                 scene context per sample. `image_size` is then the crop/tile size.
    resize_mode: str = "whole_image"
    pixel_scale: float = 1.0  # native_crop: factor applied to native pixels before cropping
    train_scale_jitter: float = 0.0  # native_crop train only: scale drawn log-uniformly in [s/(1+j), s*(1+j)]
    fg_crop_prob: float = 0.7  # native_crop train/val: chance a crop is centred near a labelled object
    train_crops_per_image: int = 1  # native_crop: crops drawn per image per epoch (epoch length scales with it)
    val_crops_per_image: int = 1  # native_crop: fixed crops per val image used for per-epoch monitoring
    # Resolved at load time (not written by hand in YAML):
    sources_resolved: dict[str, DatasetSourceConfig] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.dacl10k_test_strategy not in ("carve_val_from_train", "val_only"):
            raise ValueError(
                f"data.dacl10k_test_strategy must be 'carve_val_from_train' or 'val_only', "
                f"got {self.dacl10k_test_strategy!r}"
            )
        if not (0.0 < self.dacl10k_val_fraction < 1.0):
            raise ValueError("data.dacl10k_val_fraction must be in (0, 1)")
        if self.resize_mode not in ("whole_image", "native_crop"):
            raise ValueError(f"data.resize_mode must be 'whole_image' or 'native_crop', got {self.resize_mode!r}")
        if self.pixel_scale <= 0:
            raise ValueError("data.pixel_scale must be > 0")
        if self.train_scale_jitter < 0:
            raise ValueError("data.train_scale_jitter must be >= 0")
        if not (0.0 <= self.fg_crop_prob <= 1.0):
            raise ValueError("data.fg_crop_prob must be in [0, 1]")
        if self.train_crops_per_image < 1 or self.val_crops_per_image < 1:
            raise ValueError("data.train_crops_per_image / val_crops_per_image must be >= 1")
        if self.image_size < 512:
            raise ValueError(
                f"data.image_size={self.image_size} is below the 512px floor. A prior audit "
                "(see CLAUDE.md) traced a too-small default resolution to destroyed crack "
                "features. Raise the size or edit this guard deliberately if you understand why."
            )


@dataclass
class ModelConfig:
    name: str = "smp_efficientnet_b0"  # registry key
    encoder_name: Optional[str] = None  # only used by smp-backed models
    # smp decoder. FPN is the best CNN decoder in the dacl10k paper (FPN + EfficientNet); DeepLabV3+ was the other.
    decoder: str = "fpn"  # "fpn" | "unet" | "deeplabv3plus"
    init_mode: str = "from_scratch"  # or "task_pretrained"
    task_pretrained_checkpoint: Optional[str] = None
    # SegFormer only: also load the checkpoint's decode head (not just the encoder); the classifier is always fresh.
    load_decode_head: bool = True
    # SegFormer only: also transfer the trained classifier (requires identical class count AND order).
    load_classifier: bool = False
    dropout: float = 0.2
    num_classes: int = len(MVP_CLASSES)
    thin_structure_classes: list[str] = field(default_factory=lambda: list(THIN_STRUCTURE_CLASSES))

    def __post_init__(self) -> None:
        if self.decoder not in ("fpn", "unet", "deeplabv3plus"):
            raise ValueError(f"model.decoder must be 'fpn'/'unet'/'deeplabv3plus', got {self.decoder!r}")
        if self.init_mode not in ("from_scratch", "task_pretrained"):
            raise ValueError(f"model.init_mode must be 'from_scratch'/'task_pretrained', got {self.init_mode!r}")
        if self.init_mode == "task_pretrained" and not self.task_pretrained_checkpoint:
            raise ValueError("model.init_mode='task_pretrained' requires model.task_pretrained_checkpoint")


@dataclass
class LossConfig:
    focal_weight: float = 1.0
    dice_weight: float = 1.0
    boundary_weight: float = 0.0  # opt-in; applied only to thin_structure_classes
    dice_variant: str = "tversky"  # "dice" or "tversky"
    per_class_alpha: dict[str, float] = field(default_factory=dict)  # focal alpha, per class
    per_class_gamma: dict[str, float] = field(default_factory=dict)  # focal gamma, per class
    default_alpha: float = 0.25
    default_gamma: float = 2.0
    tversky_alpha: float = 0.3  # weight on false positives
    tversky_beta: float = 0.7  # weight on false negatives (favor recall, matches thin/rare classes)


@dataclass
class TrainingConfig:
    max_epochs: int = 30
    warmup_epochs: int = 2
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 0.01
    frozen_epochs: int = 0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "val_macro_dice"
    device: str = "cuda"
    amp: bool = True
    grad_accum_steps: int = 1
    checkpoint_dir: str = "runs"
    log_every_n_steps: int = 50  # console progress line during training epochs

    def __post_init__(self) -> None:
        if self.weight_decay <= 0.0:
            raise ValueError(
                "training.weight_decay must be > 0. A prior SegFormer script used weight_decay=0 "
                "and is flagged in CLAUDE.md as a plausible overfitting contributor — set it "
                "deliberately if you really want 0."
            )


@dataclass
class AugmentationConfig:
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    shift_scale_rotate_p: float = 0.5
    shift_limit: float = 0.05
    scale_limit: float = 0.1
    rotate_limit: int = 15
    blur_p: float = 0.3
    blur_sigma_min: float = 0.4  # a prior script's sigma floor of 0.1 was a frequent no-op; see CLAUDE.md
    blur_sigma_max: float = 2.5
    color_jitter_p: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift_limit: int = 5  # kept small: Rust/Efflorescence are partly color-defined classes
    shadow_p: float = 0.2  # random polygonal shadows (orthophotos have cast shadows / uneven illumination)
    shadow_intensity_min: float = 0.3
    shadow_intensity_max: float = 0.7


@dataclass
class WandbConfig:
    entity: str = "SET_ME"
    project: str = "SET_ME"
    run_name: Optional[str] = None
    mode: str = "online"  # "online" | "offline" | "disabled"
    log_images_every_n_epochs: int = 5
    num_logged_images: int = 8

    def validate_ready(self) -> None:
        if self.entity == "SET_ME" or self.project == "SET_ME":
            raise ValueError(
                "wandb.entity/wandb.project are still the 'SET_ME' placeholder. Set them in "
                "your experiment YAML (see configs/experiments/README or CLAUDE.md) before "
                "training — this is deliberate, not a bug."
            )


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        classes = self.data.mvp_classes
        if self.model.num_classes != len(classes):
            raise ValueError(f"model.num_classes={self.model.num_classes} but data.mvp_classes has {len(classes)} entries {classes}")
        if len(set(classes)) != len(classes):
            raise ValueError(f"data.mvp_classes contains duplicates: {classes}")
        for where, names in (
            ("model.thin_structure_classes", self.model.thin_structure_classes),
            ("loss.per_class_alpha", list(self.loss.per_class_alpha)),
            ("loss.per_class_gamma", list(self.loss.per_class_gamma)),
        ):
            unknown = [n for n in names if n not in classes]
            if unknown:
                raise ValueError(f"{where} names unknown class(es) {unknown}; data.mvp_classes = {classes}")
