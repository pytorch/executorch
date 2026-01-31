# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
from math import prod
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_node_arg
from executorch.backends.arm._passes.decompose_sum_pass import DecomposeSumPass
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.backends.arm._passes.size_adjust_input_pass import SizeAdjustInputPass
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.exir.backend.utils import WhyNoPartitionReporter
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def get_meandim_decomposition(op) -> tuple:
    if op in (exir_ops.edge.aten.mean.dim, exir_ops.edge.aten.mean.default):
        return (
            exir_ops.edge.aten.sum.dim_IntList,
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.mul.Tensor,
        )
    if op in (torch.ops.aten.mean.dim, torch.ops.aten.mean.default):
        return (
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.full.default,
            torch.ops.aten.mul.Tensor,
        )
    raise RuntimeError(f"Can't get meandim decomposition for op {op}")


def get_avgpool(op):
    if op in (exir_ops.edge.aten.mean.dim, exir_ops.edge.aten.mean.default):
        return exir_ops.edge.aten.avg_pool2d.default
    if op in (torch.ops.aten.mean.dim, torch.ops.aten.mean.default):
        return torch.ops.aten.avg_pool2d.default
    raise RuntimeError(f"Can't get meandim decomposition for op {op}")


def get_view(op):
    if op in (exir_ops.edge.aten.mean.dim, exir_ops.edge.aten.mean.default):
        return exir_ops.edge.aten.view_copy.default
    if op in (torch.ops.aten.mean.dim, torch.ops.aten.mean.default):
        return torch.ops.aten.reshape.default
    raise RuntimeError(f"Can't get meandim decomposition for op {op}")


def get_quantization(op):
    """Returns quant and dequant op of same type (per_channel/ tensor) as op if op is a dequant node, None otherwise."""
    if op in DQ_OPS:
        # Input of op can be placeholder, can't use that to get quant node directly.
        quant_type_index = DQ_OPS.index(op)
        return Q_OPS[quant_type_index], op
    return None


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

    _passes_required_after: Set[Type[ExportPass]] = {
        ComputeConstantOpsAOTPass,
        DecomposeSumPass,
        SizeAdjustInputPass,
    }

    def __init__(self, graph_module, tosa_spec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph_module = graph_module
        self._tosa_spec = tosa_spec
        # Lazy import to avoid circular dependency with operator_support
        from executorch.backends.arm.operator_support.pool_2d_support import (
            AvgPool2dSupported,
        )

        self._avg_pool_checker = AvgPool2dSupported(
            self._tosa_spec, WhyNoPartitionReporter()
        )

    def call_operator(self, op, args, kwargs, meta):
        if op not in (
            exir_ops.edge.aten.mean.dim,
            torch.ops.aten.mean.dim,
            exir_ops.edge.aten.mean.default,
            torch.ops.aten.mean.default,
        ) or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        x = get_node_arg(args, 0)
        input_shape = list(x.data.shape)
        output_shape = list(meta["val"].shape)
        dims_to_reduce = get_node_arg(args, 1, range(len(input_shape)))
        if dims_to_reduce is None:
            dims_to_reduce = range(len(input_shape))
        dims_to_reduce = [dim % len(input_shape) for dim in dims_to_reduce]
        dims_to_reduce = [dim for dim in dims_to_reduce if input_shape[dim] != 1]

        dtype = meta["val"].dtype
        view_op = get_view(op)

        # Reshape to 4D
        if len(input_shape) != 4:
            new_shape = copy(input_shape)

            while len(new_shape) < 4:
                new_shape.insert(0, 1)
                dims_to_reduce = [dim + 1 for dim in dims_to_reduce]

            while len(new_shape) > 4:
                i = new_shape.pop(0)
                new_shape[0] = new_shape[0] * i
                dims_to_reduce = [dim - 1 for dim in dims_to_reduce]

            x = super().call_operator(view_op, (x, new_shape), {}, meta, True)
            x = self._maybe_insert_q_dq_after(x, meta)

        # Reduce (h,w) dims by avg pool if possible
        x, dims_to_reduce = self._reduce_by_average_pool(op, x, dims_to_reduce, meta)

        # Reshape back to 5D if necessary
        if len(input_shape) > 4:
            original_dims = input_shape[0:-3]
            temp_shape = list(x.data.shape)[1:]
            temp_shape = original_dims + temp_shape
            dims_to_reduce = [dim + len(original_dims) - 1 for dim in dims_to_reduce]

            x = super().call_operator(view_op, (x, temp_shape), {}, meta, True)
            x = self._maybe_insert_q_dq_after(x, meta)
        # Reduce remaining dims by sum
        x = self._reduce_by_sum(op, x, dims_to_reduce, meta, dtype)

        # Reshape to correct output shape if necessary
        if list(x.data.shape) != output_shape:
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
            full_op,
            ([1] * len(output_shape), 1 / N),
            {"dtype": dtype, "device": input_node.data.device},
            meta,
            True,
        )
        if (quant_ops := get_quantization(input_node.node.target)) is not None:
            # Insert Q and DQ nodes after full op.
            # Since the value of full is known, we can compute quant params such that dq(q_max_value)
            q_op, dq_op = quant_ops
            qmax = input_node.node.args[4]
            full_quant_args = (
                1 / (N * qmax),  # Scale to map qmax to 1/N
                0,  # Zero point
                *input_node.node.args[3:],
            )
            q_args = (full, *full_quant_args)
            full = super().call_operator(
                q_op,
                q_args,
                kwargs={},
                meta=meta,
                updated=True,
            )
            dq_args = (full, *full_quant_args)
            full = super().call_operator(
                dq_op, dq_args, kwargs={}, meta=meta, updated=True
            )

            # Insert Q and DQ nodes after sum op.
            # Scale needs to be adjusted with N, since it was computed on data after the division with N.
            sum_quant_args = (input_node.node.args[1] * N, *input_node.node.args[2:])
            q_args = (sum, *sum_quant_args)
            sum = super().call_operator(
                q_op,
                q_args,
                kwargs={},
                meta=meta,
                updated=True,
            )
            dq_args = (sum, *sum_quant_args)
            sum = super().call_operator(
                dq_op, dq_args, kwargs={}, meta=meta, updated=True
            )

        return super().call_operator(mul_op, (sum, full), {}, meta, True)

    def _reduce_by_average_pool(self, op, input_node, dims, meta):
        dims_to_reduce_by_avgpool = [dim for dim in dims if dim >= 2]
        if len(dims_to_reduce_by_avgpool) == 0:
            return input_node, dims

        dims_to_reduce_by_sum = [dim for dim in dims if dim < 2]

        avgpool_op = get_avgpool(op)
        input_shape = input_node.data.size()

        stride = [1, 1]
        if dims_to_reduce_by_avgpool in ([2, 3], [3, 2]):
            kernel_size = [input_shape[2], input_shape[3]]
        elif dims_to_reduce_by_avgpool == [3]:
            kernel_size = [1, input_shape[3]]
        elif dims_to_reduce_by_avgpool == [2]:
            kernel_size = [input_shape[2], 1]
        else:
            raise RuntimeError(
                f"Bad dims {dims_to_reduce_by_avgpool} for {op} decomposition of mean_dim."
            )

        args = (input_node, kernel_size, stride)

        avg_pool_node = self._graph_module.graph.create_node(
            "call_function", avgpool_op, args
        )
        is_supported = self._avg_pool_checker.is_node_tosa_supported(
            avg_pool_node, self._tosa_spec
        )

        if is_supported:
            out = super().call_operator(avgpool_op, args, {}, meta, True)
            out = self._maybe_insert_q_dq_after(out, meta)
            return (
                out,
                dims_to_reduce_by_sum,
            )

        else:
            return input_node, dims

    def _maybe_insert_q_dq_after(self, op, meta):
        """If the input node of op is a dequant node, insert a q-dq pair after op with identical quantization parameters."""

        if len(op.node.all_input_nodes) > 1:
            raise ValueError(
                f"Expected one input to {op.node}, got inputs {op.node.all_input_nodes}"
            )
        input_node = op.node.all_input_nodes[0]
        if (quant_ops := get_quantization(input_node.target)) is not None:
            q_op, dq_op = quant_ops
            quant_args = list(input_node.args[1:])
            q_args = (op, *quant_args)
            out = super().call_operator(
                q_op,
                q_args,
                kwargs={},
                meta=meta,
                updated=True,
            )
            dq_args = (out, *quant_args)
            return super().call_operator(
                dq_op, dq_args, kwargs={}, meta=meta, updated=True
            )
        else:
            return op
