# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


@register_fake_tosa_op(
    "GATHER(Tensor values, Tensor indices) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def GATHER(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Expected signature (per TOSA):
      values:  [N, K, C]  (rank 3)
      indices: [N, W]     (rank 2, int32)
      output:  [N, W, C]  (rank 3)
    """
    tosa_spec = get_context_spec()

    # ---- dtype constraints ----
    if indices.dtype != torch.int32:
        raise TosaValueError(
            f"indices must be int32, got {indices.dtype}",
            op="GATHER",
        )

    allowed_values_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float16,
        torch.float32,
        torch.bfloat16,
    )
    if values.dtype not in allowed_values_dtypes:
        raise TosaValueError(
            "values must be one of int8/int16/int32/float16/float32/bfloat16; "
            f"got {values.dtype}",
            op="GATHER",
        )

    if values.dtype in (torch.int8, torch.int16, torch.int32):
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integers",
                op="GATHER",
            )
    else:
        # Support in FP profile, or INT profile via quantization
        if not (tosa_spec.support_float() or tosa_spec.support_integer()):
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support float",
                op="GATHER",
            )

    # ---- rank/shape constraints ----
    if values.dim() != 3:
        raise TosaValueError(
            f"values must be rank-3 [N,K,C], got shape={tuple(values.shape)}",
            op="GATHER",
        )
    if indices.dim() != 2:
        raise TosaValueError(
            f"indices must be rank-2 [N,W], got shape={tuple(indices.shape)}",
            op="GATHER",
        )
    if values.shape[0] != indices.shape[0]:
        raise TosaValueError(
            "batch mismatch: values.shape[0] != indices.shape[0] "
            f"({values.shape[0]} != {indices.shape[0]})",
            op="GATHER",
        )

    # output: [N, W, C]
    return torch.empty(
        (values.shape[0], indices.shape[1], values.shape[2]),
        dtype=values.dtype,
    )
