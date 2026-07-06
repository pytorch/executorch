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
    """
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
