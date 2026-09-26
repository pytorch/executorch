#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""Shared GGUF -> MLX qparam repack for the MLX-native lowering path.

Used when ``ET_MLX_EMIT_DIRECT_GGUF=0``: the GGUF blob is unpacked and repacked
into MLX affine qparams at export time instead of being consumed directly by
fused Metal kernels.

Q4_K / Q5_K / Q6_K repack identically -- they differ only in bit width and in
whether the merged group size can land below what MLX supports -- so each is
described by a :class:`GgufMlxFormat` and shares one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from executorch.backends.mlx.builder.op_helpers import (
    emit_quantized_biases,
    to_mlx_qparams,
)
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import Slot
from torch.fx.node import Node

# MLX's affine kernels only implement group sizes 32 / 64 / 128.
_MLX_MIN_GROUP_SIZE = 32

# Adjacent sub-blocks with an identical scale (and min) are merged into a larger
# group when lossless, up to this size.
_MAX_GROUP_SIZE = 128


@dataclass(frozen=True)
class GgufMlxFormat:
    """Per-format parameters for the MLX-native repack path."""

    ggml_type: str
    bits: int
    # Q6_K's native sub-blocks are 16 wide -- below _MLX_MIN_GROUP_SIZE -- so a
    # weight whose sub-blocks do not merge losslessly cannot use this path at
    # all and the caller has to fall back to the fused kernels. Q4_K / Q5_K are
    # natively 32, so they always clear the minimum.
    may_fall_below_min_group: bool = False

    @property
    def key_prefix(self) -> str:
        """Prefix for the constants this format emits, e.g. ``q4_k`` -> ``q4k``."""
        return self.ggml_type.replace("_", "")


Q4_K = GgufMlxFormat("q4_k", 4)
Q5_K = GgufMlxFormat("q5_k", 5)
Q6_K = GgufMlxFormat("q6_k", 6, may_fall_below_min_group=True)


def repack_mlx(
    P: MLXProgramBuilder,
    weight_node: Node,
    fmt: GgufMlxFormat,
    scale_dtype: Optional[torch.dtype] = None,
) -> Optional[Tuple[Slot, Slot, Slot, int]]:
    """Unpack a raw GGUF blob and repack it into MLX qparam constants.

    Returns ``(packed_slot, scales_slot, biases_slot, group_size)``, where
    ``group_size`` is the merged size (32, 64 or 128). Returns ``None`` only for
    a format with ``may_fall_below_min_group`` whose weight does not merge to an
    MLX-supported group size, so the caller can fall back to fused kernels.

    ``scale_dtype`` sets the dtype of the emitted scales/biases constants; pass
    the activation dtype so MLX ``quantized_matmul`` does not promote (a bf16
    activation with f16 scales, or vice versa, promotes to float32).

    Once this path is committed to, the raw blob is released so it does not sit
    alongside the packed constants for the rest of the build. It is deliberately
    kept when returning ``None``: the fused-kernel fallback reads those bytes.
    """
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor

    weight_target = P.get_placeholder_target(weight_node)
    cached = P.repack_cache.get(weight_target)
    if cached is not None:
        # Second consumer of a shared weight (e.g. a tied embedding / lm_head).
        # The raw blob is already gone, so it must not be read again.
        return cached

    _, raw = P.get_placeholder_target_and_tensor(weight_node)
    intx = ExportableGGUFTensor.from_raw(
        raw, fmt.ggml_type
    ).to_intx_unpacked_to_int8_tensor(
        max_group_size=_MAX_GROUP_SIZE, scale_dtype=scale_dtype
    )
    del raw

    group_size = int(intx.block_size[-1])
    if group_size < _MLX_MIN_GROUP_SIZE:
        if fmt.may_fall_below_min_group:
            # Falling back to the fused kernels, which consume the raw blob.
            return None
        raise ValueError(
            f"{fmt.ggml_type}: merged group size {group_size} is below MLX's "
            f"minimum of {_MLX_MIN_GROUP_SIZE}"
        )

    P.release_placeholder_tensor(weight_node)
    qdata, scale, zero_point = intx.qdata, intx.scale, intx.zero_point
    del intx  # drop the tensor-subclass wrapper; keep only the fields we need
    packed, biases = to_mlx_qparams(qdata, scale, zero_point, fmt.bits)
    del qdata  # the (N, K) int8 is no longer needed once packed

    key = f"{weight_target}_{fmt.key_prefix}"
    packed_slot = P.make_or_get_constant(f"{key}_packed", packed)
    scales_slot = P.make_or_get_constant(f"{key}_scales", scale)
    # A symmetric format (zero-point 0, e.g. Q6_K) needs no serialized biases:
    # emit_quantized_biases computes them as -scale * 2^(bits-1) in the init
    # chain instead, and falls back to a constant when the zero-point is not 0.
    biases_slot = emit_quantized_biases(
        P, key, scale, zero_point, fmt.bits, biases, scales_slot
    )

    result = (packed_slot, scales_slot, biases_slot, group_size)
    P.repack_cache[weight_target] = result
    return result
