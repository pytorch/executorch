# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

# For BI case
torch_softmax = (
    torch.ops.aten.softmax.int,
    torch.ops.aten._safe_softmax.default,
    torch.ops.aten.log_softmax.int,
)
# For MI case
edge_softmax = (
    exir_ops.edge.aten._softmax.default,
    exir_ops.edge.aten._log_softmax.default,
)
log_softmax = (torch.ops.aten.log_softmax.int, exir_ops.edge.aten._log_softmax.default)


def _get_logsoftmax_ops(op) -> tuple:
    """
    Returns the (log_op, sub_op, amax_op, expo_op, sum_op, reciprocal_op), where the ops depends on if
    the softmax op is an aten or edge op.
    """
    if op in edge_softmax:
        return (
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.amax.default,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.sum.dim_IntList,
            exir_ops.edge.aten.reciprocal.default,
            exir_ops.edge.aten.mul.Tensor,
        )
    if op in torch_softmax:
        return (
            torch.ops.aten.log.default,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.amax.default,
            torch.ops.aten.exp.default,
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.reciprocal.default,
            torch.ops.aten.mul.Tensor,
        )
    raise RuntimeError(f"Can't get logsoftmax decomposition ops for op {op}")


class DecomposeSoftmaxPass(ExportPass):
    """
    This pass decomposes log_softmax or softmax into more primitive ops.
    Example:
        %op1 = amax(x)
        %op2 = sub(x, %op1)
        %op3 = exp(%op2)
        %op4 = sum(%op3, dim)
        %op5 = reciprocal(%op4)
        %op6 = mul(%op3, %op5)
        (in logsoftmax case: %op7 = log(%op6))
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in torch_softmax + edge_softmax:
            return super().call_operator(op, args, kwargs, meta)
        log_op, sub_op, max_op, exp_op, sum_op, reciprocal_op, mul_op = (
            _get_logsoftmax_ops(op)
        )
        _input = args[0]
        dim = [args[1]]
        op1 = super().call_operator(max_op, (_input, dim, True), {}, meta)
        op2 = super().call_operator(sub_op, (_input, op1), {}, meta)
        op3 = super().call_operator(exp_op, (op2,), {}, meta)
        op4 = super().call_operator(sum_op, (op3, dim, True), {}, meta)
        op5 = super().call_operator(reciprocal_op, (op4,), {}, meta)
        op6 = super().call_operator(mul_op, (op3, op5), {}, meta)
        if op in log_softmax:
            op6 = super().call_operator(log_op, (op6,), {}, meta)
        return op6
