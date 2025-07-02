# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops

# For MI case
edge_acosh_op = exir_ops.edge.aten.acosh.default


class DecomposeAcoshPass(ArmPass):
    """
    Decomposes acosh to supported TOSA-operations.
    This decomposition is based on the mathematical identity:
        acosh(x) = log(x + sqrt((x-1)(x+1))
    """

    def call_operator(self, op, args, kwargs, meta, updated=False):

        if op is not edge_acosh_op:
            return super().call_operator(op, args, kwargs, meta, updated)

        log_op, sqrt_op, mul_op, sub_op, add_op, add_op_scalar = (
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.sqrt.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sub.Scalar,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.add.Scalar,
        )

        x = args[0]

        # (x-1)(x+1)
        sub = super().call_operator(sub_op, (x, 1.0), {}, meta, True)
        add = super().call_operator(add_op_scalar, (x, 1.0), {}, meta, True)
        mul = super().call_operator(mul_op, (sub, add), {}, meta, True)

        # sqrt((x-1)(x+1))
        sqrt = super().call_operator(sqrt_op, (mul,), {}, meta, True)

        # x + sqrt((x-1)(x+1))
        add = super().call_operator(add_op, (x, sqrt), {}, meta, True)

        # out = ln(x + sqrt((x-1)(x+1))
        out = super().call_operator(log_op, (add,), {}, meta, True)

        return out
