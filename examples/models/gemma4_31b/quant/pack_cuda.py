# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA packer: torchao quantized tensors → CUDA runtime format.

Converts ``Int4Tensor`` to ``Int4TilePackedTo4dTensor`` (tinygemm) and
passes ``IntxUnpackedToInt8Tensor`` through unchanged (AOTI fuses
the dequantize-matmul pattern).

The backend-agnostic ``pack_model`` dispatcher lives in ``pack.py``.
"""

import json

import torch
import torch.nn as nn

from .pack import ModulePackerFn, pack_model  # noqa: F401


# ---------------------------------------------------------------------------
# Low-level converters


def pack_int4_for_cuda(
    weight: torch.Tensor,
    device: str = "cuda",
) -> nn.Parameter:
    """Convert an ``Int4Tensor`` to ``Int4TilePackedTo4dTensor`` for tinygemm.

    Unpacks nibbles, pads to tinygemm alignment, tile-packs via CUDA kernel,
    and builds the combined scale_and_zero tensor.

    TODO: replace with ``Int4TilePackedTo4dTensor.from_int4_tensor()`` once
    that's upstreamed to torchao.
    """
    from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
        Int4TilePackedTo4dTensor,
    )
    from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
    from torchao.utils import find_multiple

    original_shape = weight.shape
    N, K = original_shape
    gs = weight.block_size[-1]
    inner_k_tiles = 8

    # Unpack Int4Tensor nibbles to int32
    p = weight.qdata.to(torch.uint8)
    low = (p & 0x0F).to(torch.int32)
    high = ((p >> 4) & 0x0F).to(torch.int32)
    int_data = torch.stack([low, high], dim=-1).reshape(N, K)

    # Scale/zero: Int4Tensor stores (K//gs, N), transpose to (N, K//gs)
    scale = weight.scale.t().contiguous()
    zero = weight.zero_point.t().contiguous()

    # Pad to tinygemm alignment
    K_padded = find_multiple(K, 1024)
    N_padded = find_multiple(N, 8)
    if K_padded != K or N_padded != N:
        int_data = torch.nn.functional.pad(int_data, (0, K_padded - K, 0, N_padded - N))
        n_groups_padded = K_padded // gs
        n_groups_orig = K // gs
        scale = torch.nn.functional.pad(
            scale, (0, n_groups_padded - n_groups_orig, 0, N_padded - N)
        )
        zero = torch.nn.functional.pad(
            zero, (0, n_groups_padded - n_groups_orig, 0, N_padded - N)
        )

    int_data = int_data.to(device)
    scale = scale.to(device)
    zero = zero.to(device)

    # Convert zero-point convention: tinygemm uses zp_tg = (8 - zp_std) * scale
    tinygemm_zero = (8 - zero.float()) * scale.float()

    # Tinygemm nibble convention: even=HIGH, odd=LOW
    int_data_u8 = (int_data[:, ::2] << 4 | int_data[:, 1::2]).to(torch.uint8)
    packed_weight = torch.ops.aten._convert_weight_to_int4pack(
        int_data_u8.contiguous(), inner_k_tiles
    )

    scale_and_zero = pack_tinygemm_scales_and_zeros(
        scale.to(torch.bfloat16), tinygemm_zero.to(torch.bfloat16), torch.bfloat16
    )

    subclass = Int4TilePackedTo4dTensor(
        qdata=packed_weight,
        scale_and_zero=scale_and_zero,
        block_size=[1, gs],
        shape=torch.Size(original_shape),
    )
    return nn.Parameter(subclass, requires_grad=False)


# ---------------------------------------------------------------------------
# Per-module packers


def pack_linear_for_cuda(module: nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Pack a quantized ``nn.Linear`` for CUDA."""
    from torchao.quantization import IntxUnpackedToInt8Tensor
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    w = weights["weight"]
    if isinstance(w, Int4Tensor):
        # Pack on CUDA (required by _convert_weight_to_int4pack), move back
        # to CPU for assembly. The model moves to CUDA later at runtime.
        packed = pack_int4_for_cuda(w, device="cuda")
        module.weight = nn.Parameter(packed.detach().to("cpu"), requires_grad=False)
        torch.cuda.empty_cache()
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
            "Int4TilePackedTo4dTensor does not implement the embedding op."
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
    """Load a quantized safetensors file and pack for CUDA."""
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
