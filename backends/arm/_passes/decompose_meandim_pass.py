# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_node_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def get_meandim_decomposition(op) -> tuple:
    if op == exir_ops.edge.aten.mean.dim:
        return (
            exir_ops.edge.aten.sum.dim_IntList,
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.mul.Tensor,
        )
    if op == torch.ops.aten.mean.dim:
        return (
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.full.default,
            torch.ops.aten.mul.Tensor,
        )
    raise RuntimeError(f"Can't get meandim decomposition for op {op}")


class DecomposeMeanDimPass(ExportPass):
    """
    This pass decomposes meandim into a sum and mul node.

    Example:
        y = mean_dim(x, dim, keepdim)
    Becomes:
        sum = sum.dim_IntList(x, dim, keepdim)
        y = mul(sum, 1/N)
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in (exir_ops.edge.aten.mean.dim, torch.ops.aten.mean.dim):
            return super().call_operator(op, args, kwargs, meta)

        x = get_node_arg(args, 0)
        dim = get_node_arg(args, 1)
        keepdim = get_node_arg(args, 2, False)

        # if dim == [-1, -2], mean.dim can be
        # decomposed to avg_pool2d. This is handled by ConvertMeanDimToAveragePool.
        if dim == [-1, -2]:
            # Simply return the mean.dim operator for future decomposition.
            return super().call_operator(op, args, kwargs, meta)

        shape = meta["val"].size()
        dtype = meta["val"].dtype
        input_shape = x.data.size()
        N = 1
        for d in dim:
            N *= input_shape[d]

        sum_op, full_op, mul_op = get_meandim_decomposition(op)

        sum = super().call_operator(sum_op, (x, dim, keepdim), {}, meta)
        full = super().call_operator(
            full_op, ([1] * len(shape), 1 / N), {"dtype": dtype}, meta
        )
        return super().call_operator(mul_op, (sum, full), {}, meta)
