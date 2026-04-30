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

from typing import Callable

import torch
import torch.nn as nn

from .serialize import CanonicalQuantizedWeight

# Packer signature: receives the module + a dict of its quantized weights
# (keyed by attribute name, e.g., {"weight": CQW}), modifies module in-place.
ModulePackerFn = Callable[[nn.Module, dict[str, CanonicalQuantizedWeight]], None]


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

    for fqn, tensor in unquantized.items():
        pack_one(model, fqn, tensor, packers)

    # Group quantized weights by parent module so packers that need
    # multiple weights at once (e.g., FusedMoEExperts with w1 + w2)
    # receive them in a single call.
    from collections import defaultdict

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


def pack_one(
    model: nn.Module,
    fqn: str,
    value: CanonicalQuantizedWeight | torch.Tensor,
    packers: dict[type, ModulePackerFn],
) -> None:
    """Pack a single weight into ``model``.

    If ``value`` is a ``CanonicalQuantizedWeight``, dispatches to the
    packer for the parent module's type. If it's a plain tensor, assigns
    directly as a parameter or buffer.
    """
    parts = fqn.rsplit(".", 1)
    parent_fqn = parts[0] if len(parts) > 1 else ""
    attr = parts[-1]
    parent = model.get_submodule(parent_fqn) if parent_fqn else model

    if isinstance(value, CanonicalQuantizedWeight):
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
