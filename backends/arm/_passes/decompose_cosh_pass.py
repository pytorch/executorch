# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.backends.arm._passes.match_arg_dtype_pass import MatchArgDtypePass
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

# For MI case
edge_cosh = exir_ops.edge.aten.cosh.default


class DecomposeCoshPass(ArmPass):
    """
    This pass replaces the cosh operator with a sequence of TOSA-equivalent operations that
    compute the hyperbolic cosine using the formula:

        cosh(x) = 0.5 * (e^x + e^(-x))

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        InsertTableOpsPass,
        MatchArgRanksPass,
        ReplaceScalarWithTensorByProfilePass,
        MatchArgDtypePass,
    }

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op is not edge_cosh:
            return super().call_operator(op, args, kwargs, meta, updated)

        is_quantized = (
            len(meta.data.get("input_qparams", {})) > 0
            and len(meta.data.get("output_qparams", {})) > 0
        )
        if is_quantized:
            # If quantized, node should be replace by table op
            return super().call_operator(op, args, kwargs, meta)

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
