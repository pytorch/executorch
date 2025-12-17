# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.convert_int64_const_ops_to_int32 import (
    ConvertInt64ConstOpsToInt32Pass,
)
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_scatter_ops = (exir_ops.edge.aten.select_scatter.default,)
aten_scatter_ops = (torch.ops.aten.select_scatter.default,)


def get_select_scatter_decomposition(op) -> tuple:
    if op in edge_scatter_ops:
        return (
            exir_ops.edge.aten.arange.start_step,
            exir_ops.edge.aten.eq.Scalar,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.expand_copy.default,
            exir_ops.edge.aten.unsqueeze_copy.default,
            exir_ops.edge.aten.view_copy.default,
        )
    if op in aten_scatter_ops:
        return (
            torch.ops.aten.arange.start_step,
            torch.ops.aten.eq.Scalar,
            torch.ops.aten.where.self,
            torch.ops.aten.expand_copy.default,
            torch.ops.aten.unsqueeze_copy.default,
            torch.ops.aten.view_copy.default,
        )

    raise RuntimeError(f"Can't get select_scatter decomposition for op {op}")


class DecomposeSelectScatterPass(ArmPass):
    """select_scatter is decomposed into other ops during export, however this is only
    suppported for the fp profile and for the int profile we need to decompose it here.

    The decomposition is as follows:
    - Build a boolean mask the size of x
        eq(view(arange(0, dim_size), mask_shape), index)
    - Broadcast source to x
        expand(unsqueeze(source, dim), shape)
    - Route the updated slice while keeping the untouched lanes
        where(mask, expanded_source, x)

    This reflects the decomposition for the fp profile implemented in torch._refs
    """

    _passes_required_after: Set[Type[ExportPass]] = {
        ReplaceScalarWithTensorByProfilePass,
        ConvertInt64ConstOpsToInt32Pass,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_scatter_ops + aten_scatter_ops):
            return super().call_operator(op, args, kwargs, meta, updated=False)

        (
            arange_op,
            eq_op,
            where_op,
            expand_op,
            unsqueeze_op,
            view_op,
        ) = get_select_scatter_decomposition(op)

        input_tensor = args[0]
        src_tensor = args[1]
        dim = int(args[2])
        index = int(args[3])

        shape = input_tensor.data.size()
        rank = len(shape)
        dim = dim % rank if dim < 0 else dim
        dim_size = shape[dim]
        if index < 0:
            index = index + dim_size

        mask_shape = [1] * rank
        mask_shape[dim] = -1

        arange_node = super().call_operator(
            arange_op,
            (0, dim_size, 1),
            {},
            meta,
            updated=False,
        )

        view_node = super().call_operator(
            view_op,
            (arange_node, mask_shape),
            {},
            meta,
            updated=False,
        )

        mask_node = super().call_operator(
            eq_op,
            (view_node, index),
            {},
            meta,
            updated=False,
        )

        unsqueeze_node = super().call_operator(
            unsqueeze_op,
            (src_tensor, dim),
            {},
            meta,
            updated=False,
        )

        expand_node = super().call_operator(
            expand_op,
            (unsqueeze_node, shape),
            {},
            meta,
            updated=False,
        )

        where_node = super().call_operator(
            where_op,
            (mask_node, expand_node, input_tensor),
            {},
            meta,
            updated=True,
        )

        return where_node
