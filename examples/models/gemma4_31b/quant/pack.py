# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic model packing: canonical weights → runtime model.

``pack_model`` walks a model's quantized weights, groups them by parent
module, and dispatches to per-module packer functions. Each backend
(``pack_cuda.py``, future ``pack_metal.py``) provides its own packers dict
mapping module types to packer functions.

Pure logic — no file I/O, no backend imports.
"""

from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn

from .serialize import CanonicalQuantizedWeight

# Packer signature: receives the module + a dict of its quantized weights
# (keyed by attribute name, e.g., {"weight": CQW}), modifies module in-place.
ModulePackerFn = Callable[[nn.Module, dict[str, CanonicalQuantizedWeight]], None]


def _assign_unquantized(model: nn.Module, unquantized: dict[str, torch.Tensor]) -> None:
    """Assign plain (unquantized) tensors to model parameters and buffers."""
    model_sd_keys = set(model.state_dict().keys())
    for fqn, tensor in unquantized.items():
        if fqn not in model_sd_keys:
            continue
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
        attr_name = parts[-1]
        if isinstance(getattr(parent, attr_name, None), nn.Parameter):
            setattr(parent, attr_name, nn.Parameter(tensor, requires_grad=False))
        else:
            parent.register_buffer(attr_name, tensor)


def pack_model(
    model: nn.Module,
    quantized: dict[str, CanonicalQuantizedWeight],
    unquantized: dict[str, torch.Tensor],
    packers: dict[type, ModulePackerFn],
) -> None:
    """Pack canonical weights into ``model`` using the given packers.

    Groups quantized weights by their parent module, then dispatches to the
    appropriate per-module packer based on the module's type. Models with
    custom module types (e.g., ``FusedMoEExperts``) extend ``packers``.

    Pure logic — no file I/O, no backend dependency.
    """

    _assign_unquantized(model, unquantized)

    module_weights: dict[str, dict[str, CanonicalQuantizedWeight]] = defaultdict(dict)
    for fqn, cw in quantized.items():
        parts = fqn.rsplit(".", 1)
        parent_fqn = parts[0] if len(parts) > 1 else ""
        attr = parts[-1]
        module_weights[parent_fqn][attr] = cw

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
