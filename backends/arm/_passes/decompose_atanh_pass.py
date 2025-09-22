# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


edge_atanh = exir_ops.edge.aten.atanh.default  # MI case


def _get_atanh_ops(op):
    """Return the primitive ops required.."""
    if op is not edge_atanh:
        raise RuntimeError(f"Can't decompose atanh for op {op}")
    return (
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.mul.Scalar,
        exir_ops.edge.aten.add.Scalar,
        exir_ops.edge.aten.reciprocal.default,
        exir_ops.edge.aten.log.default,
        exir_ops.edge.aten.neg.default,
    )


class DecomposeAtanhPass(ArmPass):
    """
    Decomposes the atanh operator into primitive ops.
    atanh(x) = 0.5 * log((1 + x) / (1 - x))
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op is not edge_atanh:
            return super().call_operator(op, args, kwargs, meta, updated=False)

        ops = _get_atanh_ops(op)
        (
            op_mul_tensor,
            op_mul_scalar,
            op_add_scalar,
            op_reciprocal,
            op_log,
            op_neg,
        ) = ops

        x = args[0]

        nom = super().call_operator(op_add_scalar, (x, 1.0), {}, meta, updated=True)

        neg_x = super().call_operator(op_neg, (x,), {}, meta, updated=True)
        denom = super().call_operator(
            op_add_scalar, (neg_x, 1.0), {}, meta, updated=True
        )
        recip = super().call_operator(op_reciprocal, (denom,), {}, meta, updated=True)

        log_input = super().call_operator(
            op_mul_tensor, (nom, recip), {}, meta, updated=True
        )
        log = super().call_operator(op_log, (log_input,), {}, meta, updated=True)

        return super().call_operator(op_mul_scalar, (log, 0.5), {}, meta, updated=True)
