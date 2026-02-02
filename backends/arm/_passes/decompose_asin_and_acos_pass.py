# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from math import pi
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.convert_full_like_to_full_pass import (
    ConvertFullLikeToFullPass,
)
from executorch.backends.arm._passes.decompose_div_pass import DecomposeDivPass
from executorch.backends.arm._passes.decompose_sqrt_pass import DecomposeSqrtPass
from executorch.backends.arm._passes.match_arg_dtype_pass import MatchArgDtypePass
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

# For MI case
edge_asin_op = (exir_ops.edge.aten.asin.default,)
edge_acos_op = (exir_ops.edge.aten.acos.default,)


def get_decomposition(op) -> tuple:
    if op in (edge_asin_op + edge_acos_op):
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
            exir_ops.edge.aten.neg.default,
        )

    raise RuntimeError(f"Can't get decomposition for op {op}")


class DecomposeAsinAndAcosPass(ArmPass):
    """
    This pass decomposes asin and acos into a rational approximation for small values
    and a transformed rational approximation for large values.

    The decomposition is based on the following mathematical identities:
        if abs(x) < 0.5:
            asin(x) = x + P(x^2) / Q(x^2)
            acos(x) = π/2 - asin(x)
        else:
            asin(x) = π/2 - 2 * (s + s^3 * Q(z) / P(z))
            acos(x) = 2 * (s + s^3 * Q(z) / P(z))
    where P and Q are polynomials defined in the function and s is the square root of z.

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        DecomposeSqrtPass,
        DecomposeDivPass,
        ConvertFullLikeToFullPass,
        MatchArgRanksPass,
        MatchArgDtypePass,
        ReplaceScalarWithTensorByProfilePass,
    }

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

    def _combine_branches(
        self,
        bool_op,
        bool_args: tuple[torch.Tensor, float],
        branches: tuple[torch.Tensor, torch.Tensor],
        meta: dict[str, str],
    ) -> torch.Tensor:
        where_op = exir_ops.edge.aten.where.self
        mask = super().call_operator(bool_op, bool_args, {}, meta, True)
        branch_true, branch_false = branches
        return super().call_operator(
            where_op, (mask, branch_true, branch_false), {}, meta, True
        )

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_asin_op + edge_acos_op):
            return super().call_operator(op, args, kwargs, meta)

        is_quantized = (
            len(meta.data.get("input_qparams", {})) > 0
            and len(meta.data.get("output_qparams", {})) > 0
        )
        if is_quantized:
            # If quantized, node should be replace by table op
            return super().call_operator(op, args, kwargs, meta)

        logging.info(
            f"Approximating {op}. This may introduce small numerical errors. For details, see {__file__}."
        )
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
            neg_op,
        ) = get_decomposition(op)

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

        asin_unsigned = self._combine_branches(
            gt_op, (x_abs, half), (asin_large, asin_small), meta
        )

        # Handle x < 0
        negated_asin = super().call_operator(neg_op, (asin_unsigned,), {}, meta, True)
        asin = self._combine_branches(
            lt_op, (x, zero), (negated_asin, asin_unsigned), meta
        )

        if op in edge_acos_op:
            # If x <= 0.5: acos(x) = pi/2 - asin(x)
            const_tensor = super().call_operator(
                full_like_op, (x, pi_over_2), {}, meta, True
            )
            acos_small = super().call_operator(
                sub_op, (const_tensor, asin), {}, meta, True
            )
            # If x > 0.5, acos(x) = 2 * (s + s^3 * Q(z) / P(z)) = t2
            acos = self._combine_branches(gt_op, (x, half), (t2, acos_small), meta)
            return acos

        return asin
