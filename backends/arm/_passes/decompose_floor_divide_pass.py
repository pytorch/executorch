# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.decompose_div_tensor_mode import (
    DecomposeDivTensorModePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_floor_divide_ops = (exir_ops.edge.aten.floor_divide.default,)
aten_floor_divide_ops = (torch.ops.aten.floor_divide.default,)


def get_floor_divide_decomposition(op) -> tuple:
    """
    Returns the decomposition of the given aten.floor_div operation into
    its equivalent TOSA-supported operations

    This handles both edge dialect ops and core PyTorch ops. The decomposition strategy
    is:
        floor_div(x, y) → div_tensor_mode(x, y, rounding_mode="floor")

    Returns:
        A tuple (div_op,) corresponding to the appropriate operator overload for the input op.

    Raises:
        RuntimeError: If the provided operator is not a supported floor_divide variant.
    """

    if op in edge_floor_divide_ops:
        return (
            exir_ops.edge.aten.div.Tensor_mode,
            exir_ops.edge.aten.full_like.default,
        )
    if op in aten_floor_divide_ops:
        return (
            torch.ops.aten.div.Tensor_mode,
            torch.ops.aten.full_like.default,
        )

    raise RuntimeError(f"Can't get floor_div decomposition for op {op}")


class DecomposeFloorDividePass(ArmPass):
    """
    Decomposes aten.floor_divide into aten.div.Tensor_mode with rounding_mode="floor".
    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeDivTensorModePass}

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_floor_divide_ops + aten_floor_divide_ops):
            return super().call_operator(op, args, kwargs, meta, updated=False)

        (div_op, full_op) = get_floor_divide_decomposition(op)

        input = args[0]
        other = args[1]

        if isinstance(other, int):
            other = super().call_operator(
                full_op, (input, other), {}, meta, updated=False
            )

        div_node = super().call_operator(
            div_op, (input, other), {"rounding_mode": "floor"}, meta, updated=True
        )

        return div_node
