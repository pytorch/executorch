# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class CastToInt32Pass(ArmPass):
    """Casts the input to int32 if it is not already and casts back the output to the original input dtype."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = {
        exir_ops.edge.aten.bitwise_left_shift.Tensor,
        exir_ops.edge.aten.bitwise_right_shift.Tensor,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.targeted_ops:
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
