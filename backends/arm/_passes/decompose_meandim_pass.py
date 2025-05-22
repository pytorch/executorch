# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import prod

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_node_arg
from executorch.exir.dialects._ops import ops as exir_ops


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


def get_avgpool(op):
    if op == exir_ops.edge.aten.mean.dim:
        return exir_ops.edge.aten.avg_pool2d.default
    if op == torch.ops.aten.mean.dim:
        return torch.ops.aten.avg_pool2d.default
    raise RuntimeError(f"Can't get meandim decomposition for op {op}")


def get_view(op):
    if op == exir_ops.edge.aten.mean.dim:
        return exir_ops.edge.aten.view_copy.default
    if op == torch.ops.aten.mean.dim:
        return torch.ops.aten.view_copy.default
    raise RuntimeError(f"Can't get meandim decomposition for op {op}")


class DecomposeMeanDimPass(ArmPass):
    """
    Decomposes a meandim into avg_pool and/or sum + mul (1/N) depending on which dims the mean is taken for:
        h,w -> avg_pool
        n,c -> sum + mul(1/N)
    For rank < 4, the input is first reshaped to 4D by padding with dim=1 from the left.

    Example:
        x = mean_dim(x, (0,2), keepdim=False) # x = (c,h,w)
    Becomes:
        x = view_copy.default(x, new_shape=(1,c,h,w)) # Reshape to work with avg_pool
        x = avg_pool2d.default(x, kernel=(1,w), stride=(1,1)) # Reduce w with avg_pool
        x = sum.dim_IntList(x, dim=1, keepdims=True) # Reduce c with sum
        x = mul.Tensor(x, 1/c) # Divide by number of channels to get mean
        x = view_copy.default(x, new_shape=(h)) # Squeeze dims since keepdims = False
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in (exir_ops.edge.aten.mean.dim, torch.ops.aten.mean.dim):
            return super().call_operator(op, args, kwargs, meta)

        x = get_node_arg(args, 0)
        input_shape = x.data.size()
        output_shape = meta["val"].size()
        dims_to_reduce = get_node_arg(args, 1)
        dims_to_reduce = [dim % len(input_shape) for dim in dims_to_reduce]

        dtype = meta["val"].dtype
        view_op = get_view(op)

        if len(input_shape) > 4:
            raise NotImplementedError(
                f"{op} with rank > 4 is currently not supported for the TOSA backend."
            )

        # Unsqueeze to 4D
        if len(input_shape) < 4:
            pad_n = 4 - len(input_shape)
            new_shape = [1] * pad_n + list(input_shape)
            dims_to_reduce = [dim + pad_n for dim in dims_to_reduce]

            x = super().call_operator(view_op, (x, new_shape), {}, meta, True)

        # Reduce (h,w) by avg pool
        dims_to_reduce_by_avgpool = [dim for dim in dims_to_reduce if dim >= 2]
        x = self._reduce_by_average_pool(op, x, dims_to_reduce_by_avgpool, meta)

        # Reduce (n, c) by reduce sum
        dims_to_reduce_by_sum = [dim for dim in dims_to_reduce if dim < 2]
        x = self._reduce_by_sum(op, x, dims_to_reduce_by_sum, meta, dtype)

        # Reshape to correct output shape if necessary
        if x.data.size() != output_shape:
            x = super().call_operator(view_op, (x, output_shape), {}, meta, True)

        return x

    def _reduce_by_sum(self, op, input_node, dims, meta, dtype):
        if len(dims) == 0:
            return input_node

        input_shape = input_node.data.size()
        output_shape = meta["val"].size()
        N = prod((n for i, n in enumerate(input_shape) if i in dims))
        sum_op, full_op, mul_op = get_meandim_decomposition(op)

        sum = super().call_operator(sum_op, (input_node, dims, True), {}, meta, True)
        full = super().call_operator(
            full_op, ([1] * len(output_shape), 1 / N), {"dtype": dtype}, meta, True
        )
        return super().call_operator(mul_op, (sum, full), {}, meta, True)

    def _reduce_by_average_pool(self, op, input_node, dims, meta):
        if len(dims) == 0:
            return input_node

        avgpool_op = get_avgpool(op)
        input_shape = input_node.data.size()

        stride = [1, 1]
        if dims in ([2, 3], [3, 2]):
            kernel_size = [input_shape[2], input_shape[3]]
        elif dims == [3]:
            kernel_size = [1, input_shape[3]]
        elif dims == [2]:
            kernel_size = [input_shape[2], 1]
        else:
            raise RuntimeError(f"Bad dims {dims} for {op} decomposition of mean_dim.")

        return super().call_operator(
            avgpool_op, (input_node, kernel_size, stride), {}, meta, True
        )
