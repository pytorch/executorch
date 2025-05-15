# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def _get_sum_decomp(op):
    match op:
        case exir_ops.edge.aten.sum.dim_IntList:
            return (
                exir_ops.edge.aten.view_copy.default,
                exir_ops.edge.aten.sum.dim_IntList,
            )
        case torch.ops.aten.sum.dim_IntList:
            return (torch.ops.aten.view_copy.default, torch.ops.aten.sum.dim_IntList)
        case _:
            raise RuntimeError("Unvalid op in DecomposeSumPass")


class DecomposeSumPass(ExportPass):
    """
    In Pytorch, the default behaviour of for example Tensor.sum is to squeeze the
    dimension that is summed (keep_dim = False). However, in TOSA, REDUCE_SUM always
    preserves the rank of the input (keep_dim = True). To get a 1-1 mapping in the sum
    lowering, normalize the keep_dim = False case to keep_dim = True and lower the rank
    with a view op.

    Since TOSA can only reduce one dimension at a time, multiple dims are additionally
    unrolled into multiple ops.

    Original:
        sum((dim_1, dim_2), keep_dim = False) -> squeezed_shape
    After pass:
        sum(dim_1, keep_dim = True) -> unsqueezed_shape
        sum(dim_2, keep_dim = True) -> unsqueezed_shape
        view(shape = squeezed_shape) -> squeezed_shape
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in [
            exir_ops.edge.aten.sum.dim_IntList,
            torch.ops.aten.sum.dim_IntList,
        ]:
            return super().call_operator(op, args, kwargs, meta)

        match len(args):
            case 3:
                (
                    input_node,
                    dims,
                    keepdims,
                ) = args
            case 2:
                (
                    input_node,
                    dims,
                ) = args
                keepdims = False
            case _:
                raise ValueError(f"Invalid number of arguments ({len(args)}) provided.")

        view_op, sum_op = _get_sum_decomp(op)

        for dim in dims:
            input_node = super().call_operator(
                sum_op, (input_node, dim, True), kwargs, meta
            )

        if not keepdims:
            shape = list(meta["val"].size())
            input_node = super().call_operator(
                view_op, (input_node, shape), kwargs, meta
            )

        return input_node
