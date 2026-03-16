# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.decompose_div_tensor_mode import (
    DecomposeDivTensorModePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass
from torch._ops import OpOverload

Op = OpOverload | EdgeOpOverload

_decomposition_ops: Dict[Op, tuple[Op, Op, Op]] = {
    exir_ops.edge.aten.remainder.Scalar: (
        exir_ops.edge.aten.div.Tensor_mode,
        exir_ops.edge.aten.mul.Scalar,
        exir_ops.edge.aten.sub.Tensor,
    ),
    torch.ops.aten.remainder.Tensor: (
        torch.ops.aten.div.Tensor_mode,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.sub.Tensor,
    ),
    torch.ops.aten.remainder.Scalar: (
        torch.ops.aten.div.Tensor_mode,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.sub.Tensor,
    ),
    exir_ops.edge.aten.remainder.Tensor: (
        exir_ops.edge.aten.div.Tensor_mode,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.sub.Tensor,
    ),
}


class DecomposeRemainderPass(ArmPass):
    """
    Decompose the remainder operation into primitive arithmetic:
        remainder(x, y) -> x - floor_div(x, y) * y
    where floor_div(x, y) == div(x, y, rounding_mode="floor").
    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeDivTensorModePass}

    def call_operator(self, op, args, kwargs, meta, updated=False):
        supported_ops = (
            exir_ops.edge.aten.remainder.Scalar,
            exir_ops.edge.aten.remainder.Tensor,
            torch.ops.aten.remainder.Scalar,
            torch.ops.aten.remainder.Tensor,
        )
        if op not in supported_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        div_op, mul_op, sub_op = _decomposition_ops[op]
        x, y = args[0], args[1]

        floor_div = super().call_operator(
            div_op, (x, y), {"rounding_mode": "floor"}, meta, updated=True
        )
        product = super().call_operator(mul_op, (floor_div, y), {}, meta, updated=True)
        return super().call_operator(sub_op, (x, product), {}, meta, updated=True)
