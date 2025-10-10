# Copyright 2025 Arm Limited and/or its affiliates.
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
from executorch.exir.dialects._ops import ops as exir_ops


@register_fake_tosa_op(
    "MATMUL(Tensor input1, Tensor input2) -> Tensor",  # schema
    (
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
    ),  # target TOSA specifications
)
def MATMUL(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    tosa_spec = get_context_spec()
    """Performs matrix multiplication on two input tensors.
    Additionally validates TOSA constraints of a MATMUL op.
    """
    if x1.dtype != x2.dtype:
        raise TosaValueError(
            f"Input tensors must have the same dtype, got {x1.dtype} and {x2.dtype}",
            op="MATMUL",
        )
    if x1.dtype in (torch.int8, torch.int16):
        if not tosa_spec.support_integer():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support integers", op="MATMUL"
            )
        else:
            dtype = torch.int32
    elif x1.dtype in (torch.float16, torch.float32):
        if not tosa_spec.support_float():
            raise TosaValueError(
                f"TOSA spec {tosa_spec} doesn't support float", op="MATMUL"
            )
        else:
            # float16 supports float16 accumulation as well
            dtype = torch.float32
    else:
        raise TosaValueError(
            f"Input tensors must be of type int8, float16 or float32, got {x1.dtype}",
            op="MATMUL",
        )

    aten_fake_tensor = exir_ops.edge.aten.bmm.default(x1, x2)

    return torch.empty_like(aten_fake_tensor, dtype=dtype)
