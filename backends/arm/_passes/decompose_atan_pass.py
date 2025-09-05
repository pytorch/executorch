# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from math import pi

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops


edge_atan = exir_ops.edge.aten.atan.default  # MI case


def _get_atan_ops(op):
    """Return the primitive ops required.."""
    if op is not edge_atan:
        raise RuntimeError(f"Can't decompose atan for op {op}")

    return (
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.mul.Scalar,
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.add.Scalar,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.abs.default,
        exir_ops.edge.aten.gt.Scalar,
        exir_ops.edge.aten.reciprocal.default,
        exir_ops.edge.aten.where.self,
        exir_ops.edge.aten.neg.default,
    )


class DecomposeAtanPass(ArmPass):
    """Decomposes the atan operator into a rational (Padé) approximation."""

    def _rational_approximation(self, z, ops, meta):
        """Creates a (2,1) Padé approximation for atan(x) on [-1, 1]."""

        op_mul, op_mul_scalar, op_add, op_add_scalar, _, _, _, op_recip, _, _ = ops

        # Coefficients calculated using minimax on the interval [-1, 1].
        a1 = 0.3529666667
        a2 = -0.0287666667
        b1 = 0.6863

        z2 = super().call_operator(op_mul, (z, z), {}, meta, updated=True)
        z4 = super().call_operator(op_mul, (z2, z2), {}, meta, updated=True)

        num1 = super().call_operator(op_mul_scalar, (z2, a1), {}, meta, updated=True)
        num2 = super().call_operator(op_mul_scalar, (z4, a2), {}, meta, updated=True)
        num = super().call_operator(op_add_scalar, (num1, 1.0), {}, meta, updated=True)
        num = super().call_operator(op_add, (num, num2), {}, meta, updated=True)

        den1 = super().call_operator(op_mul_scalar, (z2, b1), {}, meta, updated=True)
        den = super().call_operator(op_add_scalar, (den1, 1.0), {}, meta, updated=True)

        inv_den = super().call_operator(op_recip, (den,), {}, meta, updated=True)

        prod = super().call_operator(op_mul, (num, inv_den), {}, meta, updated=True)
        return super().call_operator(op_mul, (z, prod), {}, meta, updated=True)

    def call_operator(self, op, args, kwargs, meta):
        if op is not edge_atan:
            return super().call_operator(op, args, kwargs, meta, updated=False)

        logging.info(
            f"Approximating atan. This may introduce small numerical errors. For details, see {__file__}."
        )

        ops = _get_atan_ops(op)
        (
            _,
            op_mul_scalar,
            _,
            op_add_scalar,
            op_sub,
            op_abs,
            op_gt,
            op_recip,
            op_where,
            op_neg,
        ) = ops

        x = args[0]

        # |x| > 1 is reduced to [0, 1] using atan(x) = pi/2 - atan(1/x) and atan(-x) = -atan(x).

        abs_x = super().call_operator(op_abs, (x,), {}, meta, updated=True)
        mask_hi = super().call_operator(op_gt, (abs_x, 1.0), {}, meta, updated=True)

        inv_x = super().call_operator(op_recip, (abs_x,), {}, meta, updated=True)
        z = super().call_operator(
            op_where, (mask_hi, inv_x, abs_x), {}, meta, updated=True
        )

        atan_z = self._rational_approximation(z, ops, meta)

        zero_tensor = super().call_operator(
            op_mul_scalar, (x, 0.0), {}, meta, updated=True
        )
        half_pi_tensor = super().call_operator(
            op_add_scalar, (zero_tensor, pi / 2), {}, meta, updated=True
        )

        diff = super().call_operator(
            op_sub, (half_pi_tensor, atan_z), {}, meta, updated=True
        )
        atan_abs = super().call_operator(
            op_where, (mask_hi, diff, atan_z), {}, meta, updated=True
        )

        mask_pos = super().call_operator(op_gt, (x, 0.0), {}, meta, updated=True)
        neg_val = super().call_operator(op_neg, (atan_abs,), {}, meta, updated=True)

        return super().call_operator(
            op_where, (mask_pos, atan_abs, neg_val), {}, meta, updated=True
        )
