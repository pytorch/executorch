# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops


# For MI case
edge_addmm = exir_ops.edge.aten.addmm.default
# For BI case
aten_addmm = torch.ops.aten.addmm.default


def get_ops(op):
    """Returns the appropriate operator functions based on the input operator."""
    if op == edge_addmm:
        return (
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.add.Tensor,
        )
    elif op == aten_addmm:
        return (
            torch.ops.aten.mm.default,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.add.Tensor,
        )
    else:
        raise ValueError(f"Unsupported operator: {op}")


class DecomposeAddmmPass(ArmPass):
    """Decomposes the addmm operator into tensor multiplication and addition."""

    def call_operator(self, op, args, kwargs, meta):
        if op not in [edge_addmm, aten_addmm]:
            return super().call_operator(op, args, kwargs, meta)

        input, mat1, mat2 = args
        beta = kwargs.get("beta", 1.0)
        alpha = kwargs.get("alpha", 1.0)

        mul_op, mul_scalar_op, add_op = get_ops(op)

        mul = super().call_operator(mul_op, (mat1, mat2), {}, meta, updated=True)
        mul_alpha = super().call_operator(
            mul_scalar_op, (mul, alpha), {}, meta, updated=True
        )

        input_beta = super().call_operator(
            mul_scalar_op, (input, beta), {}, meta, updated=True
        )

        return super().call_operator(
            add_op, (mul_alpha, input_beta), {}, meta, updated=True
        )
