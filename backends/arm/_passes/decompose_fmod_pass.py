# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops

exir_op = (exir_ops.edge.aten.fmod.Tensor,)
aten_op = (torch.ops.aten.fmod.Tensor,)


def _get_decomposition(op) -> tuple:
    if op in exir_op:
        return (
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.ceil.default,
            exir_ops.edge.aten.floor.default,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.lt.Tensor,
            exir_ops.edge.aten.full_like.default,
        )
    if op in aten_op:
        return (
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.ceil.default,
            torch.ops.aten.floor.default,
            torch.ops.aten.where.self,
            torch.ops.aten.lt.Tensor,
            torch.ops.aten.full_like.default,
        )

    raise Exception(f"Unable to get decomposition for {op}")


class DecomposeFmodPass(ArmPass):
    """
    Decomposes fmod operator according to the following formula:
        fmod(x, y) = x - x.div(y, rounding_mode=truncated) * y
    """

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in (exir_op + aten_op):
            return super().call_operator(op, args, kwargs, meta, updated)

        sub_op, div_op, mul_op, ceil_op, floor_op, where_op, lt_op, full_like_op = (
            _get_decomposition(op)
        )

        x, y = args

        div = super().call_operator(div_op, (x, y), {}, meta, True)

        floor_round = super().call_operator(floor_op, (div,), {}, meta, True)
        ceil_round = super().call_operator(ceil_op, (div,), {}, meta, True)

        # Create a mask to determine which values are negative
        # and use it to select the appropriate rounding method
        # If the value is negative, use ceil, otherwise use floor
        zeros = super().call_operator(full_like_op, (div, 0.0), {}, meta, True)
        mask = super().call_operator(lt_op, (div, zeros), {}, meta, True)

        rounded_values = super().call_operator(
            where_op, (mask, ceil_round, floor_round), {}, meta, True
        )

        mul = super().call_operator(mul_op, (rounded_values, y), {}, meta, True)

        out = super().call_operator(sub_op, (x, mul), {}, meta, True)

        return out
