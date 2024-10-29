# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def get_var_decomposition(op) -> tuple:
    if op == exir_ops.edge.aten.var.correction:
        return (
            exir_ops.edge.aten.mean.dim,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sum.dim_IntList,
            exir_ops.edge.aten.full.default,
        )
    if op in (torch.ops.aten.var.correction, torch.ops.aten.var.dim):
        return (
            torch.ops.aten.mean.dim,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.full,
        )
    raise RuntimeError(f"Can't get var decomposition for op {op}")


class DecomposeVarPass(ExportPass):
    """
    This pass decomposes var.correction and var.dim into smaller ops (see https://pytorch.org/docs/stable/generated/torch.var.html)

    Example:
        y = var_correction(x, dim, keepdim, correction)
    Becomes:
        mean = mean(x, dim)
        diff = sub(x, mean)
        squared_diff = mul(diff, diff)
        sum = sum(squared_diff, dim)
        y = div(sum, max(0, N-correction))
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in (
            exir_ops.edge.aten.var.correction,
            torch.ops.aten.var.correction,
            torch.ops.aten.var.dim,
        ):
            return super().call_operator(op, args, kwargs, meta)
        shape = meta["val"].size()
        dtype = meta["val"].dtype
        dim = args[1] if len(args) > 1 else list(range(len(shape)))
        if op == torch.ops.aten.var.dim:
            correction = args[-2]
            keepdim = args[-1]
        else:
            correction = kwargs["correction"]
            keepdim = kwargs.get("keepdim", False)
        if not keepdim:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        input_shape = x.data.size()
        N = 1
        for d in dim:
            N *= input_shape[d]

        mean_op, diff_op, mul_op, sum_op, full_op = get_var_decomposition(op)
        mean = super().call_operator(mean_op, (x, dim, keepdim), {}, meta)
        diff = super().call_operator(diff_op, (x, mean), {}, meta)
        squared_diff = super().call_operator(mul_op, (diff, diff), {}, meta)
        sum = super().call_operator(sum_op, (squared_diff, dim, keepdim), {}, meta)
        full = super().call_operator(
            full_op,
            ([1 for _ in shape], 1 / max(0, N - correction)),
            {"dtype": dtype},
            meta,
        )
        return super().call_operator(mul_op, (sum, full), {}, meta)
