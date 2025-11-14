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

    _passes_required_after: Set[Type[ExportPass]] = {
        InsertTableOpsPass,
        MatchArgRanksPass,
        ReplaceScalarWithTensorByProfilePass,
        MatchArgDtypePass,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op is not edge_sinh:
            return super().call_operator(op, args, kwargs, meta)

        is_quantized = (
            len(meta.data.get("input_qparams", {})) > 0
            and len(meta.data.get("output_qparams", {})) > 0
        )
        if is_quantized:
            # If quantized, node should be replace by table op
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
