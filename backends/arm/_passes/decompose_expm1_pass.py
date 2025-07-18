# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


edge_expm1_ops = (exir_ops.edge.aten.expm1.default,)  # MI case


def _get_expm1_decomposition(op) -> tuple:
    """
    Returns the decomposition of the given aten.expm1 operation into
    its equivalent TOSA-supported operations

    This handles both edge dialect ops and core PyTorch ops. The decomposition strategy
    is:
        expm1(x) → where(and(ge(x, -0.35), le(x, 0.35)), {taylor_series_expansion}, (exp(x)-1))

    where {taylor_series_expansion} = x + (x^2/2) + (x^3/6) + (x^4/24)

    Returns:
        A tuple (op_pow, op_div, op_add, op_exp, op_sub, op_ge, op_where, op_le, op_and)
        corresponding to the appropriate operator overloads for the input op.

    Raises:
        RuntimeError: If the provided operator is not a supported elu variant.
    """
    if op in edge_expm1_ops:
        return (
            exir_ops.edge.aten.pow.Tensor_Scalar,
            exir_ops.edge.aten.div.Scalar,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.sub.Scalar,
            exir_ops.edge.aten.ge.Scalar,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.le.Scalar,
            exir_ops.edge.aten.logical_and.default,
        )

    raise RuntimeError(f"Can't get expm1 decomposition for op {op}")


class DecomposeExpm1Pass(ArmPass):
    """
    A transformation pass that decomposes unsupported 'aten.expm1' operations
    into a combination of supported TOSA-equivalent operations.

    Since TOSA does not provide a native expm1 operator, this pass rewrites:
        expm1(x) →  where(and(ge(x, -0.35), le(x, 0.35)), {taylor_series_expansion}, (exp(x)-1))
    where {taylor_series_expansion} = x + (x^2/2) + (x^3/6) + (x^4/24)

    Supported input ops:
        - exir_ops.edge.aten.expm1.default(x)

    These are replaced with:
        - exir_ops.edge.aten.pow.Tensor_Scalar,
        - exir_ops.edge.aten.div.Scalar,
        - exir_ops.edge.aten.add.Tensor,
        - exir_ops.edge.aten.exp.default,
        - exir_ops.edge.aten.sub.Scalar,
        - exir_ops.edge.aten.ge.Scalar,
        - exir_ops.edge.aten.where.self,
        - exir_ops.edge.aten.le.Scalar,
        - exir_ops.edge.aten.logical_and.default
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in edge_expm1_ops:
            return super().call_operator(op, args, kwargs, meta, updated=False)

        (
            op_pow,
            op_div,
            op_add,
            op_exp,
            op_sub,
            op_ge,
            op_where,
            op_le,
            op_and,
        ) = _get_expm1_decomposition(op)

        input = args[0]

        cutlo = -0.35
        cuthi = 0.35

        taylor_term_2_numerator = super().call_operator(
            op_pow, (input, 2), {}, meta, updated=False
        )
        taylor_term_3_numerator = super().call_operator(
            op_pow, (input, 3), {}, meta, updated=False
        )
        taylor_term_4_numerator = super().call_operator(
            op_pow, (input, 4), {}, meta, updated=False
        )

        taylor_term_2 = super().call_operator(
            op_div, (taylor_term_2_numerator, 2), {}, meta, updated=False
        )
        taylor_term_3 = super().call_operator(
            op_div, (taylor_term_3_numerator, 6), {}, meta, updated=False
        )
        taylor_term_4 = super().call_operator(
            op_div, (taylor_term_4_numerator, 24), {}, meta, updated=False
        )

        add_terms_1_2 = super().call_operator(
            op_add, (input, taylor_term_2), {}, meta, updated=False
        )
        add_term_3 = super().call_operator(
            op_add, (add_terms_1_2, taylor_term_3), {}, meta, updated=False
        )
        taylor_expansion = super().call_operator(
            op_add, (add_term_3, taylor_term_4), {}, meta, updated=False
        )

        decomp_exp = super().call_operator(op_exp, (input,), {}, meta, updated=False)
        decomp_sub = super().call_operator(
            op_sub, (decomp_exp, 1.0), {}, meta, updated=False
        )

        ge = super().call_operator(op_ge, (input, cutlo), {}, meta, updated=False)
        le = super().call_operator(op_le, (input, cuthi), {}, meta, updated=False)

        cond_and = super().call_operator(op_and, (ge, le), {}, meta, updated=False)
        where = super().call_operator(
            op_where, (cond_and, taylor_expansion, decomp_sub), {}, meta, updated=True
        )

        return where
