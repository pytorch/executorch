# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops


# For MI case
edge_sinh = exir_ops.edge.aten.sinh.default


class DecomposeSinhPass(ArmPass):
    """
    A decomposition pass that decomposes Sinh operations into a
    combination of supported TOSA-equivalent operations (MI).

    Supported input ops:
        - exir_ops.edge.aten.sinh.default

    These are decomposed into exponentials, negation, subtraction,
        and scalar multiplication.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op is not edge_sinh:
            return super().call_operator(op, args, kwargs, meta)

        x = args

        sub_op, exp_op, neg_op, mul_op = (
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.neg.default,
            exir_ops.edge.aten.mul.Scalar,
        )

        # Exponential 1
        exp1 = super().call_operator(exp_op, x, {}, meta, updated=True)

        # Exponential 2
        neg_x = super().call_operator(neg_op, x, {}, meta, updated=True)
        exp2 = super().call_operator(exp_op, (neg_x,), {}, meta, updated=True)

        # Subtraction
        sub = super().call_operator(sub_op, (exp1, exp2), {}, meta, updated=True)

        # Multiplication
        out = super().call_operator(mul_op, (sub, 0.5), {}, meta, updated=True)

        return out
