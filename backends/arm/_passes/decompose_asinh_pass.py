# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops

# For MI case
edge_asinh_op = (exir_ops.edge.aten.asinh.default,)


class DecomposeAsinhPass(ArmPass):
    """
    Decomposes asinh to supported TOSA-operations.
    This decomposition is based on the mathematical identity:
        asinh(x) = log(x + sqrt(x^2 + 1))
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in edge_asinh_op:
            return super().call_operator(op, args, kwargs, meta)

        log_op, sqrt_op, mul_op, add_op_scalar, add_op = (
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.sqrt.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.add.Scalar,
            exir_ops.edge.aten.add.Tensor,
        )

        x = args[0]

        # calculate t1 = x^2 + 1
        x2 = super().call_operator(mul_op, (x, x), {}, meta, True)
        t1 = super().call_operator(add_op_scalar, (x2, 1.0), {}, meta, True)

        # t2 = sqrt(t1)
        t2 = super().call_operator(sqrt_op, (t1,), {}, meta, True)

        # t3 = x + t2
        t3 = super().call_operator(add_op, (x, t2), {}, meta, True)

        # out = ln(t3)
        out = super().call_operator(log_op, (t3,), {}, meta, True)

        return out
