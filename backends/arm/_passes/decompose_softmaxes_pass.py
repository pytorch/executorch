# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

# For BI case
torch_softmax = (torch.ops.aten.softmax.int, torch.ops.aten.log_softmax.int)

# For MI case
edge_softmax = (
    exir_ops.edge.aten._softmax.default,
    exir_ops.edge.aten._log_softmax.default,
)

log_softmax = (torch.ops.aten.log_softmax.int, exir_ops.edge.aten._log_softmax.default)


def get_logsoftmax_ops(op) -> tuple:
    """
    Returns the the (log_op, expo_op, sum_op, reciprocal_op), where the ops depends on if
    the logsoftmax op is in exir_ops torch.ops.aten.
    """
    if op in edge_softmax:
        return (
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.sum.dim_IntList,
            exir_ops.edge.aten.reciprocal.default,
            exir_ops.edge.aten.mul.Tensor,
        )
    if op in torch_softmax:
        return (
            torch.ops.aten.log.default,
            torch.ops.aten.exp.default,
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.reciprocal.default,
            torch.ops.aten.mul.Tensor,
        )
    raise RuntimeError(f"Can't get softmax decomposition ops for op {op}")


class DecomposeSoftmaxesPass(ExportPass):
    """
    This pass decomposes log softmax or softmax into more primitive ops.

    Example:
        %op1 = exp(x)
        %op2 = sum(%op1, dim)
        %op3 = reciprocal(%op2)
        %op4 = mul(%op1, %op3)
        (in logsoftmax case: %op5 = log(%op4))
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in torch_softmax + edge_softmax:
            return super().call_operator(op, args, kwargs, meta)

        log_op, exp_op, sum_op, reciprocal_op, mul_op = get_logsoftmax_ops(op)

        _input = args[0]
        dim = [args[1]]

        op1 = super().call_operator(exp_op, (_input,), {}, meta)
        op2 = super().call_operator(sum_op, (op1, dim, True), {}, meta)
        op3 = super().call_operator(reciprocal_op, (op2,), {}, meta)
        op4 = super().call_operator(mul_op, (op1, op3), {}, meta)
        if op in log_softmax:
            op4 = super().call_operator(log_op, (op4,), {}, meta)
        return op4
