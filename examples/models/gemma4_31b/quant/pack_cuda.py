# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA packer: assign quantized weights to model modules.

Passes ``Int4Tensor`` and ``IntxUnpackedToInt8Tensor`` through as
``nn.Parameter`` without conversion.  The Int4Tensor dispatch override
(``int4_dispatch.py``) handles F.linear at runtime.

No CUDA is required for packing.  The backend-agnostic ``pack_model``
dispatcher lives in ``pack.py``.
"""

import json

import torch
import torch.nn as nn

from .pack import ModulePackerFn, pack_model  # noqa: F401


# ---------------------------------------------------------------------------
# Per-module packers


def pack_linear_for_cuda(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Assign a quantized weight to an ``nn.Linear`` module."""
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, (Int4Tensor, IntxUnpackedToInt8Tensor)):
        module.weight = nn.Parameter(w, requires_grad=False)
    else:
        raise ValueError(f"Unsupported weight type: {type(w).__name__}")


def pack_embedding_for_cuda(
    module: nn.Module, weights: dict[str, torch.Tensor]
) -> None:
    """Assign a quantized weight to an ``nn.Embedding`` (INT8 only)."""
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, Int4Tensor):
        raise ValueError(
            "Only 8-bit embedding quantization is supported on CUDA. "
            "INT4 does not implement the embedding op."
        )
    module.weight = nn.Parameter(w, requires_grad=False)


DEFAULT_CUDA_PACKERS: dict[type, ModulePackerFn] = {
    nn.Linear: pack_linear_for_cuda,
    nn.Embedding: pack_embedding_for_cuda,
}


# ---------------------------------------------------------------------------
# Load + pack (I/O wrapper)


def load_and_pack_for_cuda(
    path: str,
    model: nn.Module,
    packers: dict[type, ModulePackerFn] | None = None,
) -> None:
    """Load a quantized safetensors file and assign weights to the model."""
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import (
        unflatten_tensor_state_dict,
    )

    from .pack import pack_one

    _packers = packers or DEFAULT_CUDA_PACKERS
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        all_keys = list(f.keys())
        tensor_names = json.loads(metadata.get("tensor_names", "[]"))

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
                pack_one(model, fqn, value, _packers)

    for fqn, p in model.named_parameters():
        if p.device.type == "meta":
            raise RuntimeError(
                f"Weight '{fqn}' not found in checkpoint "
                f"(model/checkpoint version mismatch?)"
            )
