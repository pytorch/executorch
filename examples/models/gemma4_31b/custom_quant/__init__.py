# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gemma4 31B-specific quantization and packing helpers."""

import json

import torch
import torch.nn as nn

from executorch.examples.models.gemma4_31b.quant import (
    DEFAULT_CUDA_PACKERS,
    DEFAULT_MLX_PACKERS,
    ModulePackerFn,
    pack_model as _generic_pack_model,
    quantize_model as _generic_quantize_model,
)
from executorch.examples.models.gemma4_31b.quant.pack import pack_one

from .pack_vision import (  # noqa: F401
    collect_vision_state_dict,
    has_vision_keys,
    install_int8_pe_dispatch,
    quantize_vision_position_table,
    VISION_PREFIXES,
)


def quantize_model(
    model: nn.Module,
    recipe,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """Quantize Gemma4 31B and fold the vision PE table into int8 buffers."""
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is not None:
        quantize_vision_position_table(vision_tower, verbose=verbose)
    return _generic_quantize_model(model, recipe, dtype=dtype, verbose=verbose)


def _maybe_install_vision_pe_dispatch(model: nn.Module, keys) -> None:
    """Install int8 PE dispatch for vision towers referenced by state keys."""
    pe_int8_keys = [k for k in keys if k.endswith("._pet_int8")]
    if not pe_int8_keys:
        return

    seen: set[str] = set()
    for key in pe_int8_keys:
        patch_embedder_fqn = key[: -len("._pet_int8")]
        vision_tower_fqn = patch_embedder_fqn.rsplit(".", 1)[0]
        if vision_tower_fqn in seen:
            continue
        seen.add(vision_tower_fqn)
        try:
            vision_tower = model.get_submodule(vision_tower_fqn)
        except AttributeError:
            continue
        install_int8_pe_dispatch(vision_tower)


def pack_model(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    packers: dict[type, ModulePackerFn],
) -> None:
    """Pack Gemma4 31B state dict after installing vision PE dispatch."""
    _maybe_install_vision_pe_dispatch(model, state_dict.keys())
    _generic_pack_model(model, state_dict, packers)


def load_and_pack_for_cuda(
    path: str,
    model: nn.Module,
    packers: dict[type, ModulePackerFn] | None = None,
) -> None:
    """Load and pack a Gemma4 31B CUDA checkpoint with vision PE dispatch."""
    _load_and_pack(path, model, packers or DEFAULT_CUDA_PACKERS)


def load_and_pack_for_mlx(
    path: str,
    model: nn.Module,
    packers: dict[type, ModulePackerFn] | None = None,
) -> None:
    """Load and pack a Gemma4 31B MLX checkpoint with vision PE dispatch."""
    _load_and_pack(path, model, packers or DEFAULT_MLX_PACKERS)


def _load_and_pack(
    path: str,
    model: nn.Module,
    packers: dict[type, ModulePackerFn],
) -> None:
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        all_keys = list(f.keys())
        tensor_names = json.loads(metadata.get("tensor_names", "[]"))

        _maybe_install_vision_pe_dispatch(model, tensor_names)

        for name in tensor_names:
            parts = name.rsplit(".", 1)
            module_fqn = parts[0] if len(parts) > 1 else ""
            weight_name = parts[-1]
            prefix = (
                f"{module_fqn}._{weight_name}_" if module_fqn else f"_{weight_name}_"
            )
            partial = {}
            for key in all_keys:
                if key.startswith(prefix) or key == name:
                    partial[key] = f.get_tensor(key)
            result, _ = unflatten_tensor_state_dict(partial, metadata)
            for fqn, value in result.items():
                pack_one(model, fqn, value, packers)

    for fqn, param in model.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in checkpoint "
                f"(model/checkpoint version mismatch?)"
            )


__all__ = [
    "VISION_PREFIXES",
    "collect_vision_state_dict",
    "has_vision_keys",
    "install_int8_pe_dispatch",
    "load_and_pack_for_cuda",
    "load_and_pack_for_mlx",
    "pack_model",
    "quantize_model",
    "quantize_vision_position_table",
]
