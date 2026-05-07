# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Source transform: convert INT4 linears to tinygemm format.

After packing, INT4 linears hold ``IntxUnpackedToInt8Tensor`` weights
whose ``F.linear`` dispatch dequantizes to bf16 and calls cuBLAS.
This is optimal for prefill (large M, compute-bound).

``use_tinygemm_linears`` converts these to ``Int4TilePackedTo4dTensor``
(tinygemm), which is optimal for decode (M=1, bandwidth-bound).

Typical export flow::

    prefill_ep = export(model, ...)          # IntxUnpacked → dequant+cuBLAS
    use_tinygemm_linears(model)              # one-way conversion
    decode_ep = export(model, ...)           # tinygemm

Model-agnostic: operates on any ``nn.Linear`` whose weight is an
``IntxUnpackedToInt8Tensor`` with group_size < K (i.e., not per-axis).
"""

import torch
import torch.nn as nn


def _is_int4_intx(param: torch.Tensor) -> bool:
    """Check if a parameter is an IntxUnpackedToInt8Tensor holding 4-bit data.

    INT4 weights (from int4_tensor_to_intx) have bf16 zero_point with
    unsigned [0, 15] qdata.  INT8 weights have int8 zero_point with
    signed [-128, 127] qdata.
    """
    from torchao.quantization import IntxUnpackedToInt8Tensor

    if not isinstance(param, IntxUnpackedToInt8Tensor):
        return False
    return param.zero_point.dtype != torch.int8


def _intx_to_tinygemm(weight: torch.Tensor) -> torch.Tensor:
    """Convert IntxUnpackedToInt8Tensor (int4 in int8) to Int4TilePackedTo4dTensor."""
    from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
        Int4TilePackedTo4dTensor,
    )
    from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
    from torchao.utils import find_multiple

    N, K = weight.shape
    gs = weight.block_size[-1]
    inner_k_tiles = 8

    int_data = weight.qdata.to(torch.int32)
    scale = weight.scale
    zero = weight.zero_point

    K_padded = find_multiple(K, 1024)
    N_padded = find_multiple(N, 8)

    if K_padded != K or N_padded != N:
        int_data = torch.nn.functional.pad(int_data, (0, K_padded - K, 0, N_padded - N))
        ng_pad = K_padded // gs
        ng_orig = K // gs
        scale = torch.nn.functional.pad(scale, (0, ng_pad - ng_orig, 0, N_padded - N))
        zero = torch.nn.functional.pad(zero, (0, ng_pad - ng_orig, 0, N_padded - N))

    orig_device = int_data.device
    int_data = int_data.to("cuda")
    scale = scale.to("cuda")
    zero = zero.to("cuda")

    tinygemm_zero = (8 - zero.float()) * scale.float()
    u8 = (int_data[:, ::2] << 4 | int_data[:, 1::2]).to(torch.uint8)
    tg_packed = torch.ops.aten._convert_weight_to_int4pack(
        u8.contiguous(), inner_k_tiles
    )
    scale_and_zero = pack_tinygemm_scales_and_zeros(
        scale.to(torch.bfloat16), tinygemm_zero.to(torch.bfloat16), torch.bfloat16
    )

    return Int4TilePackedTo4dTensor(
        qdata=tg_packed.to(orig_device),
        scale_and_zero=scale_and_zero.to(orig_device),
        block_size=[1, gs],
        shape=torch.Size([N, K]),
    )


def use_tinygemm_linears(model: nn.Module) -> None:
    """Convert INT4 ``IntxUnpackedToInt8Tensor`` linears to tinygemm format.

    Optimal for decode (M=1). Requires CUDA for the tile-packing kernel.
    This is a one-way conversion — the original IntxUnpacked data is not
    preserved.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear) and _is_int4_intx(module.weight):
            tg = _intx_to_tinygemm(module.weight.data)
            module.weight = nn.Parameter(tg, requires_grad=False)
    torch.cuda.empty_cache()
