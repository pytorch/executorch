# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class CastIntComparisonInputsPass(ArmOpTargetedPass):
    """Cast integer comparison inputs to a lossless floating-point type."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    target_ops = {
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.ne.Tensor,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.lt.Tensor,
    }
    castable_dtypes = {torch.int8, torch.int16}

    def should_run_pass(self, graph_module: torch.fx.GraphModule) -> bool:
        tosa_spec = get_context_spec()
        return (
            tosa_spec.support_float()
            and not tosa_spec.support_integer()
            and super().should_run_pass(graph_module)
        )

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        if not all(arg.data.dtype in self.castable_dtypes for arg in args):
            return super().call_operator(op, args, kwargs, meta)

        cast_dtype = (
            torch.float16
            if all(arg.data.dtype == torch.int8 for arg in args)
            else torch.float32
        )
        casted_args = []
        for arg in args:
            casted_args.append(
                super().call_operator(
                    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                    (arg,),
                    {"dtype": cast_dtype},
                    meta,
                )
            )
        return super().call_operator(op, tuple(casted_args), kwargs, meta)
