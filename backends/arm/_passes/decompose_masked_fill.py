# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


edge_ops = (exir_ops.edge.aten.masked_fill.Scalar,)
aten_ops = (torch.ops.aten.masked_fill.Scalar,)


def _get_decomposition(op) -> tuple:
    if op in edge_ops:
        return (
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.full_like.default,
        )
    if op in aten_ops:
        return (
            torch.ops.aten.where.self,
            torch.ops.aten.full_like.default,
        )
    raise RuntimeError(f"Unable to get decomposition for op {op}")


class DecomposeMaskedFill(ArmPass):
    """
    Masked fill takes in a boolean mask, a tensor and a scalar value.
    Fills the tensor with the scalar value according to the boolean mask.
    Decomposed to a where and a full_like operator.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in (edge_ops + aten_ops):
            return super().call_operator(op, args, kwargs, meta, updated)

        x, mask, scalar = args

        where_op, full_like_op = _get_decomposition(op)

        scalar_tensor = super().call_operator(full_like_op, (x, scalar), {}, meta, True)

        return super().call_operator(
            where_op, (mask, scalar_tensor, x), kwargs, meta, True
        )
