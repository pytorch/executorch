# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm._passes.decompose_sum_pass import DecomposeSumPass
from executorch.backends.arm.tosa.specification import (
    get_context_shape_env,
    get_context_spec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata


class ConvertBoolSumPass(ArmOpTargetedPass):
    """Compute boolean sums with an int32 accumulator."""

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeSumPass}
    _MAX_EXACT_SUM = torch.iinfo(torch.int32).max
    target_ops = (
        torch.ops.aten.sum.dim_IntList,
        exir_ops.edge.aten.sum.dim_IntList,
    )
    check_allowed_to_transform = True

    def call_operator(self, op, args, kwargs, meta):
        tosa_spec = get_context_spec()
        if (
            op not in self.target_ops
            or args[0].data.dtype != torch.bool
            or not self.allowed_to_transform(meta)
            or not tosa_spec.support_integer()
            or tosa_spec.support_float()
        ):
            return super().call_operator(op, args, kwargs, meta)

        dims = args[1]
        if not dims:
            dims = range(args[0].data.dim())
        reduced_elements = math.prod(args[0].data.shape[dim] for dim in dims)
        if isinstance(reduced_elements, torch.SymInt):
            reduced_elements = (
                get_context_shape_env().bound_sympy(reduced_elements.node.expr).upper
            )
        if reduced_elements > self._MAX_EXACT_SUM:
            return super().call_operator(op, args, kwargs, meta)

        cast_op = (
            exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            if op == exir_ops.edge.aten.sum.dim_IntList
            else torch.ops.dim_order_ops._to_dim_order_copy.default
        )
        accumulator_input = super().call_operator(
            cast_op,
            (args[0],),
            {"dtype": torch.int32},
            NodeMetadata(args[0].node.meta),
            updated=True,
        )
        sum_kwargs = dict(kwargs)
        sum_kwargs["dtype"] = torch.int32
        accumulator_sum = super().call_operator(
            op,
            (accumulator_input, *args[1:]),
            sum_kwargs,
            meta,
            updated=True,
        )
        return super().call_operator(
            cast_op,
            (accumulator_sum,),
            {"dtype": meta["val"].dtype},
            meta,
            updated=True,
        )
