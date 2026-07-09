#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Q5_K -> MLX qparam repack for the legacy MLX-native lowering path.

Used when ``ET_MLX_EMIT_DIRECT_GGUF=0``: the GGUF blob is unpacked and repacked
into MLX affine 5-bit qparams at export time instead of being consumed directly
by fused Metal kernels.
"""

from __future__ import annotations

from typing import Tuple

from executorch.backends.mlx.builder.op_helpers import to_mlx_qparams
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from torch.fx.node import Node

_BITS = 5


def repack_mlx(P: MLXProgramBuilder, weight_node: Node) -> Tuple[Slot, Slot, Slot, int]:
    """Unpack a raw Q5_K blob and repack into MLX qparam constants.

    Adjacent sub-blocks with identical scale/min are merged into a larger group
    size (up to 128) when lossless, so ``group_size`` may be 32, 64, or 128.
    Returns ``(packed_slot, scales_slot, biases_slot, group_size)``.
    """
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor

    weight_target, raw = P.get_placeholder_target_and_tensor(weight_node)
    intx = ExportableGGUFTensor.from_raw(raw, "q5_k").to_intx_unpacked_to_int8_tensor(
        max_group_size=128
    )
    group_size = int(intx.block_size[-1])
    packed, biases = to_mlx_qparams(intx.qdata, intx.scale, intx.zero_point, _BITS)

    packed_slot = P.make_or_get_constant(f"{weight_target}_q5k_packed", packed)
    scales_slot = P.make_or_get_constant(f"{weight_target}_q5k_scales", intx.scale)
    biases_slot = P.make_or_get_constant(f"{weight_target}_q5k_biases", biases)
    return packed_slot, scales_slot, biases_slot, group_size
