# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_div_ops = (exir_ops.edge.aten.div.Tensor,)
aten_div_ops = (torch.ops.aten.div.Tensor, torch.ops.aten.div_.Tensor)


def get_div_decomposition(op) -> tuple:
    """
    Returns the the (reciprocal_op, mul_op), where the ops depends on if
    the div op is in exir_ops torch.ops.aten.
    """
    if op in edge_div_ops:
        return (exir_ops.edge.aten.reciprocal.default, exir_ops.edge.aten.mul.Tensor)
    if op in aten_div_ops:
        return (torch.ops.aten.reciprocal.default, torch.ops.aten.mul.Tensor)
    raise RuntimeError(f"Can't get div decomposition for op {op}")


class DecomposeDivPass(ExportPass):
    """
    This pass decomposes div into a mul and a reciprocal node.

    Example:
        y = div(a,b)
    Becomes:
        x = reciprocal(b)
        y = mul(a,x)
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_div_ops + aten_div_ops):
            return super().call_operator(op, args, kwargs, meta)

        reciprocal_op, mul_op = get_div_decomposition(op)

        numerator = args[0]
        denominator = args[1]
        reciprocal = super().call_operator(reciprocal_op, (denominator,), {}, meta)

        return super().call_operator(mul_op, (numerator, reciprocal), {}, meta)
