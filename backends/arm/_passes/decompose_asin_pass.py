# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from math import pi

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops

# For MI case
edge_asin_op = (exir_ops.edge.aten.asin.default,)


def get_asin_decomposition(op) -> tuple:
    if op in edge_asin_op:
        return (
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.sqrt.default,
            exir_ops.edge.aten.abs.default,
            exir_ops.edge.aten.sub.Scalar,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten.gt.Scalar,
            exir_ops.edge.aten.lt.Scalar,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.full_like.default,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.neg.default,
        )

    raise RuntimeError(f"Can't get asin decomposition for op {op}")


class DecomposeAsinPass(ArmPass):
    """
    This pass decomposes asin into a rational approximation for small values
    and a transformed rational approximation for large values.
    Example:
        y = asin(x)
    Becomes:
        if abs(x) < 0.5:
            y = x + P(x^2) / Q(x^2)
        else:
            y = π/2 - 2 * (s + s^3 * Q(z) / P(z))
    where P and Q are polynomials defined in the function.
    """

    def _build_polynomial(
        self, coefficients: list[float], variable: torch.Tensor, meta: dict[str, str]
    ) -> torch.Tensor:
        """
        Helper function to build polynomial from coefficients and variable.
        """
        full_like_op, add_op, mul_op_scalar, mul_op = (
            exir_ops.edge.aten.full_like.default,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.mul.Tensor,
        )
        result = super().call_operator(
            full_like_op, (variable, coefficients[0]), {}, meta, True
        )
        for coeff in coefficients[1:]:
            result = super().call_operator(
                add_op,
                (
                    result,
                    super().call_operator(
                        mul_op_scalar, (variable, coeff), {}, meta, True
                    ),
                ),
                {},
                meta,
            )
            variable = super().call_operator(
                mul_op, (variable, variable), {}, meta, True
            )
        return result

    def call_operator(self, op, args, kwargs, meta):
        logging.info(
            f"Approximating asin. This may introduce small numerical errors. For details, see {__file__}."
        )
        if op not in edge_asin_op:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        half = 0.5
        one = 1.0
        neg_half = -0.5
        two = 2.0
        pi_over_2 = pi / 2.0
        zero = 0.0
        neg_one = -1.0

        (
            mul_op,
            add_op,
            mul_op_scalar,
            sqrt_op,
            abs_op,
            sub_op_scalar,
            div_op,
            gt_op,
            lt_op,
            sub_op,
            full_like_op,
            where_op,
            neg_op,
        ) = get_asin_decomposition(op)

        # Coefficients for the rational approximation, calculated with the Minimax (Remez) method
        p_coefficients = [
            1.6666667163e-01,
            -3.2556581497e-01,
            2.0121252537e-01,
            -4.0055535734e-02,
            7.9153501429e-04,
        ]

        q_coefficients = [1.0, -2.4033949375e00, 2.0209457874e00, -6.8828397989e-01]

        x_abs = super().call_operator(abs_op, (x,), {}, meta, True)

        # Step 1: compute asin_small - rational approximation for [0,0.5]

        y = super().call_operator(mul_op, (x_abs, x_abs), {}, meta, True)
        x3 = super().call_operator(mul_op, (x_abs, y), {}, meta, True)

        P = self._build_polynomial(p_coefficients, x_abs, meta)
        Q = self._build_polynomial(q_coefficients, x_abs, meta)
        numer = super().call_operator(mul_op, (x3, P), {}, meta, True)
        r_small = super().call_operator(div_op, (numer, Q), {}, meta, True)
        asin_small = super().call_operator(add_op, (x_abs, r_small), {}, meta, True)

        # Step 2: Compute the transformed approximation for large values
        # Calculate z = -0.5 * (|x| - 1)
        tmp_ones = super().call_operator(full_like_op, (x_abs, one), {}, meta, True)
        tmp = super().call_operator(sub_op, (x_abs, tmp_ones), {}, meta, True)
        z = super().call_operator(mul_op_scalar, (tmp, neg_half), {}, meta, True)

        # Calculate s-terms
        s = super().call_operator(sqrt_op, (z,), {}, meta, True)
        s2 = super().call_operator(mul_op, (s, s), {}, meta, True)
        s3 = super().call_operator(mul_op, (s2, s), {}, meta, True)

        Pz = self._build_polynomial(p_coefficients, z, meta)
        Qz = self._build_polynomial(q_coefficients, z, meta)

        numer = super().call_operator(mul_op, (s3, Pz), {}, meta, True)
        # Calculate r_large = P(z) / Q(z)
        r_large = super().call_operator(div_op, (numer, Qz), {}, meta, True)

        # Calculate asin_large = pi/2 - 2 * (s + s^3 * Q(z) / P(z))
        t1 = super().call_operator(add_op, (s, r_large), {}, meta, True)
        t2 = super().call_operator(mul_op_scalar, (t1, two), {}, meta, True)
        diff = super().call_operator(sub_op_scalar, (t2, pi_over_2), {}, meta, True)
        tmp_neg_ones = super().call_operator(
            full_like_op, (diff, neg_one), {}, meta, True
        )
        asin_large = super().call_operator(mul_op, (diff, tmp_neg_ones), {}, meta, True)

        # Combine branches
        is_large = super().call_operator(gt_op, (x_abs, half), {}, meta, True)
        asin_unsigned = super().call_operator(
            where_op,
            (
                is_large,
                asin_large,
                asin_small,
            ),
            {},
            meta,
            True,
        )

        # Handle x < 0
        is_neg = super().call_operator(lt_op, (x, zero), {}, meta, True)
        # Compute -asin_unsigned
        negated_asin = super().call_operator(neg_op, (asin_unsigned,), {}, meta, True)
        # Combine branches for signed asin
        asin_signed = super().call_operator(
            where_op,
            (
                is_neg,
                negated_asin,
                asin_unsigned,
            ),
            {},
            meta,
            True,
        )

        return asin_signed
