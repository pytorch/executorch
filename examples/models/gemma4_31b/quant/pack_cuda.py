# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA packer: canonical → CUDA runtime format.

Provides per-module packers for the CUDA backend (INT4 via tinygemm,
INT8 via ``IntxUnpackedToInt8Tensor``) and ``load_and_pack_for_cuda``
as a convenience I/O wrapper.

The backend-agnostic ``pack_model`` dispatcher lives in ``pack.py``.
"""

import torch
import torch.nn as nn

from .pack import ModulePackerFn, pack_model  # noqa: F401
from .serialize import CanonicalQuantizedWeight, load


# ---------------------------------------------------------------------------
# Low-level: canonical → Int4TilePackedTo4dTensor (one weight at a time)


def pack_int4_for_cuda(
    cw: CanonicalQuantizedWeight,
    device: str = "cuda",
) -> nn.Parameter:
    """Convert a canonical 4-bit weight to ``Int4TilePackedTo4dTensor``.

    Pads K to a multiple of 1024 and N to a multiple of 8 (tinygemm
    requirements), nibble-packs, then tile-packs via the CUDA kernel.
    Returns an ``nn.Parameter`` wrapping the subclass tensor **on CUDA**.
    """
    from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
        Int4TilePackedTo4dTensor,
    )
    from torchao.quantization.utils import pack_tinygemm_scales_and_zeros
    from torchao.utils import find_multiple

    assert cw.config.bits == 4, f"Expected 4-bit, got {cw.config.bits}"
    assert cw.qdata.ndim == 2, (
        f"pack_int4_for_cuda requires 2D weight (nn.Linear), got {cw.qdata.ndim}D "
        f"shape {tuple(cw.qdata.shape)}."
    )

    original_shape = cw.qdata.shape
    N, K = original_shape
    gs = cw.config.group_size
    inner_k_tiles = 8

    K_padded = find_multiple(K, 1024)
    N_padded = find_multiple(N, 8)

    int_data = cw.qdata.to(torch.int32)
    if K_padded != K or N_padded != N:
        int_data = torch.nn.functional.pad(int_data, (0, K_padded - K, 0, N_padded - N))

    scale = cw.scale
    n_groups_orig = K // gs
    n_groups_padded = K_padded // gs
    if n_groups_padded != n_groups_orig or N_padded != N:
        scale = torch.nn.functional.pad(
            scale, (0, n_groups_padded - n_groups_orig, 0, N_padded - N)
        )

    if cw.zero is not None:
        zero = cw.zero
        if n_groups_padded != n_groups_orig or N_padded != N:
            zero = torch.nn.functional.pad(
                zero, (0, n_groups_padded - n_groups_orig, 0, N_padded - N)
            )
    else:
        # Symmetric: qdata is unsigned [0, 15] (shifted +8 from signed [-8, 7]).
        # Standard convention: weight = (q - zp_std) * scale, so zp_std = 8.
        zero = torch.full_like(scale, 8.0)

    int_data = int_data.to(device)
    scale = scale.to(device)
    zero = zero.to(device)

    # Convert zero from standard convention (weight = (q - zp_std) * scale)
    # to tinygemm convention (weight = (q - 8) * scale + zp_tg).
    # Derivation: (q - zp_std) * scale = (q - 8) * scale + zp_tg
    #           → zp_tg = (8 - zp_std) * scale
    tinygemm_zero = (8 - zero.to(torch.float32)) * scale.to(torch.float32)

    # Tinygemm nibble convention: even index in HIGH nibble, odd in LOW.
    # (This differs from serialize.py's _nibble_pack which uses the opposite
    # convention for on-disk storage — both are valid, they serve different
    # consumers.)
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


def pack_int8_for_cuda(
    cw: CanonicalQuantizedWeight,
) -> nn.Parameter:
    """Convert a canonical 8-bit weight to ``IntxUnpackedToInt8Tensor``.

    Unlike INT4 (which needs tinygemm tile packing), INT8 weights are stored
    unpacked. The subclass carries int8 qdata + scales and dequantizes during
    matmul — AOTI fuses the ``dequantize → mm`` pattern in the compiled graph.
    """
    from torchao.quantization import IntxUnpackedToInt8Tensor

    assert cw.config.bits == 8, f"Expected 8-bit, got {cw.config.bits}"
    assert cw.qdata.ndim == 2, f"Expected 2D weight, got {cw.qdata.ndim}D"

    N, K = cw.qdata.shape
    n_groups = K // cw.config.group_size
    scale = cw.scale.to(torch.bfloat16).reshape(N, n_groups)
    zero_point = (
        cw.zero.to(torch.int8).reshape(N, n_groups)
        if cw.zero is not None
        else torch.zeros(N, n_groups, dtype=torch.int8)
    )

    subclass = IntxUnpackedToInt8Tensor(
        qdata=cw.qdata,
        scale=scale,
        zero_point=zero_point,
        target_dtype=torch.int8,
        block_size=(1, cw.config.group_size),
        dtype=torch.bfloat16,
        activation_quantization=None,
    )
    return nn.Parameter(subclass, requires_grad=False)


def pack_linear_for_cuda(
    module: nn.Module, weights: dict[str, CanonicalQuantizedWeight]
) -> None:
    """Pack a quantized ``nn.Linear`` for CUDA.

    4-bit weights use ``Int4TilePackedTo4dTensor`` (tinygemm kernel, requires
    CUDA for packing). 8-bit weights use ``IntxUnpackedToInt8Tensor`` (AOTI
    fuses the dequantize-matmul pattern). Both stay as tensor subclasses so
    the export graph captures quantized ops.
    """
    cw = weights["weight"]
    if cw.config.bits == 4:
        packed = pack_int4_for_cuda(cw, device="cuda")
        module.weight = nn.Parameter(packed.data.to("cpu"), requires_grad=False)
        torch.cuda.empty_cache()
    elif cw.config.bits == 8:
        module.weight = pack_int8_for_cuda(cw)
    else:
        raise ValueError(f"Unsupported bit width: {cw.config.bits}")


def pack_embedding_for_cuda(
    module: nn.Module, weights: dict[str, CanonicalQuantizedWeight]
) -> None:
    """Pack a quantized ``nn.Embedding`` for CUDA.

    Uses ``IntxUnpackedToInt8Tensor`` which supports embedding gather.
    Only INT8 is supported — ``Int4TilePackedTo4dTensor`` does not
    implement the embedding op.
    """
    cw = weights["weight"]
    if cw.config.bits != 8:
        raise ValueError(
            f"Only 8-bit embedding quantization is supported on CUDA, "
            f"got {cw.config.bits}-bit."
        )
    module.weight = pack_int8_for_cuda(cw)


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
    """Read a quantized safetensors file and pack into ``model`` for CUDA.

    Thin wrapper: ``load`` + ``pack_model``.
    """
    quantized, unquantized = load(path)
    pack_model(model, quantized, unquantized, packers or DEFAULT_CUDA_PACKERS)
