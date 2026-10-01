"""Load a task-pretrained SegFormer checkpoint (e.g. a dacl10k-trained MiT) into our wrapper.

The local task-pretrained SegFormers come in two shapes, both written by an older transformers
(4.34) and therefore using its *legacy* parameter names:

1. a pickled model object — `torch.save(model)` of a `SegModel` wrapper holding a
   `SegformerForSemanticSegmentation` (e.g. `bikit_models/models/checkpoints/best_model_cpu.pth`,
   MiT-b1 with a 19-class dacl10k head);
2. a plain `state_dict` — `torch.save(model.state_dict())` from a previous run
   (e.g. `SegF_model/semseg_runs/segformer_custom_*/best_model.pth`).

Rather than hand-writing the 4.x -> 5.x rename map (`linear_c` -> `linear_projections`,
`encoder.block.i.j` -> `stages.i.blocks.j`, `attention.self.query` -> `attention.q_proj`, ...),
the tensors are written to a temporary HF-format checkpoint and handed to
`from_pretrained`, which applies transformers' own legacy conversion. Verified against a hub
checkpoint in the old format (`nvidia/segformer-b0-finetuned-ade-512-512`): zero missing and zero
unexpected keys, classifier size mismatch reported cleanly. A name-based mapping is what makes
this safe — matching by shape alone could silently swap q/k/v projections, which share shapes.

Reading the pickled variant requires unpickling, which executes the pickle's opcodes. The
original classes are NOT needed (unavailable ones are replaced by inert stubs and only the module
tree's tensors are read), but only point this at checkpoints you trust — like your own.
"""
from __future__ import annotations

import json
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# Shared by every MiT variant (b0..b5); only hidden_sizes and depths differ between them.
MIT_NUM_ATTENTION_HEADS = [1, 2, 5, 8]
MIT_SR_RATIOS = [8, 4, 2, 1]

logger = logging.getLogger(__name__)


class SegformerCheckpointError(RuntimeError):
    pass


class _Stub:
    """Stands in for a class the current environment no longer has; captures state only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class _PermissiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return getattr(__import__(module, fromlist=[name]), name)
        except Exception:
            return type(f"Stub_{name}", (_Stub,), {})


class _permissive_pickle_module:
    """Minimal `pickle_module` interface for torch.load."""

    Unpickler = _PermissiveUnpickler

    @staticmethod
    def load(f, **kwargs):
        return _PermissiveUnpickler(f, **kwargs).load()


def _tensors_from_module_tree(obj: Any) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def walk(node: Any, prefix: str) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        attrs = getattr(node, "__dict__", {})
        for container in ("_parameters", "_buffers"):
            for name, tensor in (attrs.get(container) or {}).items():
                if tensor is not None:
                    out[f"{prefix}{name}"] = tensor.detach().cpu()
        for name, child in (attrs.get("_modules") or {}).items():
            if child is not None:
                walk(child, f"{prefix}{name}.")

    walk(obj, "")
    return out


def extract_state_dict(path: str | Path) -> tuple[dict[str, torch.Tensor], str]:
    """Returns (state_dict, source_format). Accepts a plain state_dict, a `{"state_dict": ...}`
    wrapper, or a pickled model object."""
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        source_format = "state_dict"
    except Exception:
        loaded = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_permissive_pickle_module)
        source_format = "pickled_model"

    if isinstance(loaded, dict):
        state = loaded.get("state_dict", loaded)
        if not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise SegformerCheckpointError(f"{path}: dict does not hold only tensors; unsupported checkpoint layout")
        state = {k: v.detach().cpu() for k, v in state.items()}
    else:
        state = _tensors_from_module_tree(loaded)

    if not state:
        raise SegformerCheckpointError(f"{path}: no tensors found")
    return _strip_wrapper_prefix(state), source_format


def _strip_wrapper_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A `SegModel`-style wrapper nests everything under one attribute, giving
    `segformer.decode_head.*` / `segformer.segformer.encoder.*`. HF expects `decode_head.*` /
    `segformer.encoder.*`, i.e. decode_head at the root — that is the unambiguous signal."""
    prefixes = {k.split(".", 1)[0] for k in state}
    for prefix in prefixes:
        if any(k.startswith(f"{prefix}.decode_head.") for k in state):
            return {k[len(prefix) + 1 :] if k.startswith(f"{prefix}.") else k: v for k, v in state.items()}
    return state


def infer_segformer_config(state: dict[str, torch.Tensor], num_labels: int) -> SegformerConfig:
    """Derives the MiT architecture from tensor shapes, so any variant (b0..b5) works without
    being told which one it is."""
    legacy = any(".encoder.patch_embeddings." in k for k in state)
    hidden_sizes, depths, mlp_ratios, sr_ratios = [], [], [], []
    stage = 0
    while True:
        proj = f"segformer.encoder.patch_embeddings.{stage}.proj.weight" if legacy else f"segformer.stages.{stage}.patch_embeddings.proj.weight"
        if proj not in state:
            break
        hidden = state[proj].shape[0]
        hidden_sizes.append(hidden)
        block_prefix = f"segformer.encoder.block.{stage}." if legacy else f"segformer.stages.{stage}.blocks."
        block_ids = {int(k[len(block_prefix) :].split(".", 1)[0]) for k in state if k.startswith(block_prefix)}
        depths.append(max(block_ids) + 1)
        fc1 = f"{block_prefix}0.mlp.{'dense1' if legacy else 'fc1'}.weight"
        mlp_ratios.append(int(state[fc1].shape[0] // hidden) if fc1 in state else 4)
        sr = f"{block_prefix}0.attention.{'self.sr' if legacy else 'sequence_reduction.sequence_reduction'}.weight"
        sr_ratios.append(int(state[sr].shape[-1]) if sr in state else MIT_SR_RATIOS[stage])
        stage += 1

    if len(hidden_sizes) != 4:
        raise SegformerCheckpointError(f"expected 4 encoder stages, found {len(hidden_sizes)} — not a MiT/SegFormer checkpoint?")

    classifier = state.get("decode_head.classifier.weight")
    decoder_hidden_size = int(classifier.shape[1]) if classifier is not None else 256
    return SegformerConfig(
        num_encoder_blocks=4,
        depths=depths,
        hidden_sizes=hidden_sizes,
        mlp_ratios=mlp_ratios,
        sr_ratios=sr_ratios,
        num_attention_heads=MIT_NUM_ATTENTION_HEADS,
        decoder_hidden_size=decoder_hidden_size,
        num_labels=num_labels,
    )


def load_segformer_pretrained(
    net: SegformerForSemanticSegmentation,
    path: str | Path,
    load_decode_head: bool = True,
    load_classifier: bool = False,
) -> dict[str, Any]:
    """Loads `path` into an already-built `net` in place. Raises if the checkpoint's architecture
    does not match `net`, or if any non-classifier parameter fails to load.

    `load_classifier` transfers the final per-class layer too, which is only possible when the
    checkpoint has the same number of classes — and only *correct* if it used the same class
    order. It stays off by default because a silent order mismatch would scramble the classes."""
    state, source_format = extract_state_dict(path)
    ckpt_classifier = state.get("decode_head.classifier.weight")
    source_num_labels = int(ckpt_classifier.shape[0]) if ckpt_classifier is not None else None
    inferred = infer_segformer_config(state, num_labels=net.config.num_labels)

    for field in ("hidden_sizes", "depths"):
        want, got = list(getattr(net.config, field)), list(getattr(inferred, field))
        if want != got:
            raise SegformerCheckpointError(
                f"{path}: checkpoint architecture {field}={got} does not match the configured model "
                f"({field}={want}). Set model.encoder_name to the matching MiT variant "
                f"(hidden_sizes {got} corresponds to {'mit-b0' if got[0] == 32 else 'mit-b1 or larger'})."
            )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        save_file({k: v.contiguous().clone() for k, v in state.items()}, tmp_path / "model.safetensors")
        (tmp_path / "config.json").write_text(json.dumps(net.config.to_dict(), default=str))
        converted, info = SegformerForSemanticSegmentation.from_pretrained(
            tmp_path, config=net.config, ignore_mismatched_sizes=True, output_loading_info=True
        )

    mismatched = {k for k, *_ in info["mismatched_keys"]}
    blocking = [k for k in info["missing_keys"] if k not in mismatched and not k.startswith("decode_head.classifier")]
    if blocking:
        raise SegformerCheckpointError(
            f"{path}: {len(blocking)} parameter(s) could not be loaded, e.g. {sorted(blocking)[:5]}. "
            "Aborting rather than silently training a partly-random encoder."
        )

    new_state = converted.state_dict()
    skip = mismatched | ({k for k in new_state if k.startswith("decode_head.")} if not load_decode_head else set())
    classifier_keys = {k for k in new_state if k.startswith("decode_head.classifier")}
    transfer_classifier = load_classifier and load_decode_head and not (classifier_keys & mismatched)
    if not transfer_classifier:
        skip |= classifier_keys
    net.load_state_dict({k: v for k, v in new_state.items() if k not in skip}, strict=False)

    if not load_classifier and source_num_labels == net.config.num_labels:
        logger.info(
            "checkpoint has the same number of classes (%d) as this model; set model.load_classifier=true to "
            "transfer the trained classifier too — only do that if the class ORDER matches.",
            source_num_labels,
        )

    return {
        "checkpoint_path": str(path),
        "source_format": source_format,
        "source_num_labels": source_num_labels,
        "hidden_sizes": list(inferred.hidden_sizes),
        "depths": list(inferred.depths),
        "tensors_loaded": len(new_state) - len(skip),
        "tensors_in_checkpoint": len(state),
        "decode_head_loaded": load_decode_head,
        "classifier_reinitialised": not transfer_classifier,
        "unexpected_keys": sorted(info["unexpected_keys"]),
    }
