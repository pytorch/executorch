# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA packer: assign quantized weights to model modules.

Converts ``Int4Tensor`` (nibble-packed) to ``IntxUnpackedToInt8Tensor``
(int4 values unpacked to int8).  Passes ``IntxUnpackedToInt8Tensor``
(8-bit) through unchanged.

No CUDA is required for packing.  The backend-agnostic ``pack_model``
dispatcher lives in ``pack.py``.
"""

import json

import torch
import torch.nn as nn

from .pack import ModulePackerFn, pack_model  # noqa: F401


# ---------------------------------------------------------------------------
# Low-level converters


def int4_tensor_to_intx(weight: torch.Tensor) -> torch.Tensor:
    """Convert an ``Int4Tensor`` to ``IntxUnpackedToInt8Tensor``.

    Unpacks nibble-packed qdata to int8 (keeping unsigned [0, 15] values)
    and transposes scale/zero_point from Int4Tensor's ``(K//gs, N)``
    layout to ``(N, K//gs)``.  Uses bf16 zero_point to preserve the
    asymmetric offset.
    """
    from torchao.quantization import IntxUnpackedToInt8Tensor

    N, K = weight.shape
    gs = weight.block_size[-1]

    p = weight.qdata.to(torch.uint8)
    low = (p & 0x0F).to(torch.int8)
    high = ((p >> 4) & 0x0F).to(torch.int8)
    qdata = torch.stack([low, high], dim=-1).reshape(N, K)

    scale = weight.scale.t().contiguous()
    zero_point = weight.zero_point.t().contiguous()

    return IntxUnpackedToInt8Tensor(
        qdata=qdata,
        scale=scale,
        zero_point=zero_point,
        target_dtype=torch.int8,
        block_size=(1, gs),
        dtype=torch.bfloat16,
        activation_quantization=None,
    )


# ---------------------------------------------------------------------------
# Per-module packers


def pack_linear_for_cuda(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Pack a quantized ``nn.Linear`` for CUDA."""
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, Int4Tensor):
        module.weight = nn.Parameter(int4_tensor_to_intx(w), requires_grad=False)
    elif isinstance(w, IntxUnpackedToInt8Tensor):
        module.weight = nn.Parameter(w, requires_grad=False)
    else:
        raise ValueError(f"Unsupported weight type: {type(w).__name__}")


def pack_embedding_for_cuda(
    module: nn.Module, weights: dict[str, torch.Tensor]
) -> None:
    """Pack a quantized ``nn.Embedding`` for CUDA (INT8 only)."""
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

        # Stream one logical weight at a time: load its inner tensors,
        # reconstruct the subclass, pack, then release before the next.
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
