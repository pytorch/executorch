# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic model packing: quantized state dict → runtime model.

``pack_model`` walks a state dict, groups quantized weights by parent
module, and dispatches to per-module packer functions. Each backend
(``pack_cuda.py``, future ``pack_metal.py``) provides its own packers dict.
"""

from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn

# Packer signature: receives the module + a dict of its quantized weights
# (keyed by attribute name), modifies module in-place.
ModulePackerFn = Callable[[nn.Module, dict[str, torch.Tensor]], None]


def _is_quantized(value: torch.Tensor) -> bool:
    """Check if a tensor is a torchao quantized subclass."""
    from torchao.utils import TorchAOBaseTensor

    return isinstance(value, TorchAOBaseTensor)


def pack_model(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    packers: dict[type, ModulePackerFn],
) -> None:
    """Pack a state dict into ``model`` using the given packers.

    Quantized weights (torchao tensor subclasses) are grouped by parent
    module and dispatched to per-module packers. Plain tensors are assigned
    directly as parameters or buffers.

    Vision PE int8 dispatch is installed automatically. If the state dict
    carries ``*._pet_int8`` keys, the matching ``vision_tower`` submodule
    has its bf16 ``position_embedding_table`` parameter swapped for
    placeholder ``_pet_int8`` / ``_pet_scale`` buffers + the int8
    embedding-lookup monkey-patch (idempotent). This makes the load path
    symmetric with the save path (``quantize_model`` folds in the
    corresponding ``quantize_vision_position_table`` swap on the save
    side) so callers do not need to invoke ``install_int8_pe_dispatch``
    manually.
    """
    _maybe_install_vision_pe_dispatch(model, state_dict.keys())

    # Separate quantized and unquantized
    for fqn, value in state_dict.items():
        if not _is_quantized(value):
            pack_one(model, fqn, value, packers)

    # Group quantized weights by parent module
    module_weights: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for fqn, value in state_dict.items():
        if _is_quantized(value):
            parts = fqn.rsplit(".", 1)
            parent_fqn = parts[0] if len(parts) > 1 else ""
            attr = parts[-1]
            module_weights[parent_fqn][attr] = value

    for parent_fqn, weights in module_weights.items():
        module = model.get_submodule(parent_fqn) if parent_fqn else model
        packer = packers.get(type(module))
        if packer is None:
            raise ValueError(
                f"No packer registered for {type(module).__name__} at '{parent_fqn}'. "
                f"Registered types: {[t.__name__ for t in packers]}."
            )
        packer(module, weights)

    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in checkpoint "
                f"(model/checkpoint version mismatch?)"
            )

    for p in model.parameters():
        p.requires_grad_(False)


def _maybe_install_vision_pe_dispatch(model: nn.Module, keys) -> None:
    """Install the int8 PE dispatch for any vision_tower whose patch
    embedder is referenced by ``*._pet_int8`` entries in ``keys``.

    ``keys`` may be any iterable of logical FQNs (e.g. a state dict's
    keys, or the ``tensor_names`` list pulled from safetensors metadata).
    Idempotent — safe to call when dispatch is already installed.
    """
    pe_int8_keys = [k for k in keys if k.endswith("._pet_int8")]
    if not pe_int8_keys:
        return

    from .pack_vision_cuda import install_int8_pe_dispatch

    seen: set[str] = set()
    for k in pe_int8_keys:
        # Key shape: "<vision_tower_fqn>.patch_embedder._pet_int8"
        patch_embedder_fqn = k[: -len("._pet_int8")]
        vt_fqn = patch_embedder_fqn.rsplit(".", 1)[0]
        if vt_fqn in seen:
            continue
        seen.add(vt_fqn)
        try:
            vision_tower = model.get_submodule(vt_fqn)
        except AttributeError:
            # State dict references a submodule that the model lacks; let
            # the regular pack path produce the standard error.
            continue
        install_int8_pe_dispatch(vision_tower)


def pack_one(
    model: nn.Module,
    fqn: str,
    value: torch.Tensor,
    packers: dict[type, ModulePackerFn],
) -> None:
    """Pack a single weight into ``model``.

    Quantized subclass tensors are dispatched to the packer for the parent
    module's type. Plain tensors are assigned directly.
    """
    parts = fqn.rsplit(".", 1)
    parent_fqn = parts[0] if len(parts) > 1 else ""
    attr = parts[-1]
    parent = model.get_submodule(parent_fqn) if parent_fqn else model

    if _is_quantized(value):
        packer = packers.get(type(parent))
        if packer is None:
            raise ValueError(
                f"No packer registered for {type(parent).__name__} at '{parent_fqn}'. "
                f"Registered types: {[t.__name__ for t in packers]}."
            )
        packer(parent, {attr: value})
    else:
        if isinstance(getattr(parent, attr, None), nn.Parameter):
            setattr(parent, attr, nn.Parameter(value, requires_grad=False))
        else:
            parent.register_buffer(attr, value)
