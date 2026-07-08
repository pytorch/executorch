# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class CastToInt32Pass(ArmOpTargetedPass):
    """Casts the input to int32 if it is not already and casts back the output
    to the original input dtype.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    target_ops = {
        exir_ops.edge.aten.bitwise_left_shift.Tensor,
        exir_ops.edge.aten.bitwise_right_shift.Tensor,
    }

    def should_run_pass(self, graph_module: torch.fx.GraphModule) -> bool:
        tosa_spec = get_context_spec()
        if not tosa_spec.is_U55_subset:
            return False
        return super().should_run_pass(graph_module)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        new_args: list = []
        did_cast = False
        for arg in args:
            if arg.data.dtype != torch.int32:
                new_args.append(
                    super().call_operator(
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        (arg,),
                        {"dtype": torch.int32},
                        meta,
                    )
                )
                did_cast = True
            else:
                new_args.append(arg)

        output = super().call_operator(
            op,
            tuple(new_args),
            {},
            meta,
        )

        if did_cast:
            output = super().call_operator(
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                (output,),
                {"dtype": args[0].data.dtype},
                meta,
            )
        return output
