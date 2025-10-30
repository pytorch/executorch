# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_ne_ops = (exir_ops.edge.aten.ne.Tensor,)
aten_ne_ops = (torch.ops.aten.ne.Tensor, torch.ops.aten.ne_.Tensor)


def get_ne_decomposition(op) -> tuple:
    """
    Returns the decomposition of the given aten.ne operation into its equivalent
    TOSA-supported operations.

    This handles both edge dialect ops and core PyTorch ops. The decomposition strategy
    is:
        ne(x, y) -> logical_not(eq(x, y))

    Returns:
        A tuple (eq_op, logical_not_op) corresponding to the appropriate operator
        overloads for the input op.

    Raises:
        RuntimeError: If the provided operator is not a supported ne variant.
    """
    if op in edge_ne_ops:
        return (exir_ops.edge.aten.eq.Tensor, exir_ops.edge.aten.logical_not.default)
    if op in aten_ne_ops:
        return (torch.ops.aten.eq.Tensor, torch.ops.aten.logical_not.default)

    raise RuntimeError(f"Can't get ne decomposition for op {op}")


class DecomposeNotEqualPass(ArmPass):
    """
    A transformation pass that decomposes unsupported `aten.ne` operations into a
    combination of supported TOSA-equivalent operations.

    Since TOSA does not provide a native NOT_EQUAL operator, this pass rewrites:
        ne(x, y) → logical_not(eq(x, y))

    Supported input ops:
        - aten.ne.Tensor(x, y)
        - aten.ne_.Tensor(x, y)
        - exir_ops.edge.aten.ne.Tensor(x, y)

    These are replaced with:
        - aten.eq.Tensor or exir_ops.edge.aten.eq.Tensor
        - followed by aten.logical_not.default or its edge equivalent
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_ne_ops + aten_ne_ops):
            return super().call_operator(op, args, kwargs, meta)

        lhs, rhs = args

        eq_op, logical_not_op = get_ne_decomposition(op)

        eq_node = super().call_operator(eq_op, (lhs, rhs), {}, meta, updated=True)
        not_node = super().call_operator(
            logical_not_op, (eq_node,), {}, meta, updated=True
        )

        return not_node
