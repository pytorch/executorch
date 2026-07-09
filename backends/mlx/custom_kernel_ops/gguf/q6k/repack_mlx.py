#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Q6_K -> MLX qparam repack for the MLX-native lowering path.

Used when ``ET_MLX_EMIT_DIRECT_GGUF=0``: the GGUF blob is unpacked and repacked
into MLX affine 6-bit qparams at export time instead of being consumed directly
by fused Metal kernels.

Q6_K's native group size is 16, which MLX's affine kernels do not support (only
32/64/128), so the repack succeeds only when adjacent sub-blocks merge
losslessly into a group size >= 32 (see
:func:`...gguf.ExportableGGUFTensor.to_intx_unpacked_to_int8_tensor`). When it
cannot, :func:`repack_mlx` returns ``None`` so the caller falls back to the
fused kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple

from executorch.backends.mlx.builder.op_helpers import (
    emit_quantized_biases,
    to_mlx_qparams,
)
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from torch.fx.node import Node

_BITS = 6

# MLX affine kernels only support these group sizes; Q6_K's native 16 is not one.
_MIN_MLX_GROUP_SIZE = 32


def repack_mlx(
    P: MLXProgramBuilder, weight_node: Node
) -> Optional[Tuple[Slot, Slot, Slot, int]]:
    """Unpack a raw Q6_K blob and repack into MLX qparam constants.

    Adjacent sub-blocks with identical scale are merged into a larger group size
    (up to 128) when lossless. Returns ``(packed_slot, scales_slot, biases_slot,
    group_size)`` when the merged ``group_size`` is MLX-compatible (>= 32), or
    ``None`` when it is not (so the caller can fall back to fused kernels).
    """
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor

    weight_target, raw = P.get_placeholder_target_and_tensor(weight_node)
    intx = ExportableGGUFTensor.from_raw(raw, "q6_k").to_intx_unpacked_to_int8_tensor(
        max_group_size=128
    )
    group_size = int(intx.block_size[-1])
    if group_size < _MIN_MLX_GROUP_SIZE:
        return None

    packed, biases = to_mlx_qparams(intx.qdata, intx.scale, intx.zero_point, _BITS)

    packed_slot = P.make_or_get_constant(f"{weight_target}_q6k_packed", packed)
    scales_slot = P.make_or_get_constant(f"{weight_target}_q6k_scales", intx.scale)
    # Q6_K is symmetric (zero-point 0): emit_quantized_biases computes
    # biases = -scale * 2^(bits-1) in the init chain instead of serializing them.
    biases_slot = emit_quantized_biases(
        P,
        f"{weight_target}_q6k",
        intx.scale,
        intx.zero_point,
        _BITS,
        biases,
        scales_slot,
    )
    return packed_slot, scales_slot, biases_slot, group_size
