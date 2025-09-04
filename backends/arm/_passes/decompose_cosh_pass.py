# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops

# For MI case
edge_cosh = exir_ops.edge.aten.cosh.default


class DecomposeCoshPass(ArmPass):
    """
    This pass replaces the cosh operator with a sequence of TOSA-equivalent operations that
    compute the hyperbolic cosine using the formula:

        cosh(x) = 0.5 * (e^x + e^(-x))

    """

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op is not edge_cosh:
            return super().call_operator(op, args, kwargs, meta, updated)

        x = args

        exp_op, mul_op, neg_op, add_op = (
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.neg.default,
            exir_ops.edge.aten.add.Tensor,
        )

        # exp1 = e^x
        exp1 = super().call_operator(exp_op, x, {}, meta, updated=True)

        # exp2 = e^(⁻x)
        neg_x = super().call_operator(neg_op, x, {}, meta, updated=True)
        exp2 = super().call_operator(exp_op, (neg_x,), {}, meta, updated=True)

        # numer = exp1 + exp2
        numer = super().call_operator(add_op, (exp1, exp2), {}, meta, updated=True)

        # out = 0.5 * numer
        out = super().call_operator(mul_op, (numer, 0.5), {}, meta, updated=True)

        return out
