# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import get_node_arg
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def _get_ops(op):
    """Return the namespace-matched operators used by the decomposition."""
    tril_copy = getattr(exir_ops.edge.aten, "tril_copy", None)

    if op in (exir_ops.edge.aten.tril.default, getattr(tril_copy, "default", None)):
        return {
            "arange": exir_ops.edge.aten.arange.default,
            "view": exir_ops.edge.aten.view_copy.default,
            "add_scalar": exir_ops.edge.aten.add.Scalar,
            "ge_tensor": exir_ops.edge.aten.ge.Tensor,
            "where": exir_ops.edge.aten.where.self,
            "scalar_tensor": exir_ops.edge.aten.scalar_tensor.default,
            "bitwise_and": exir_ops.edge.aten.bitwise_and.Tensor,
        }

    if op in (torch.ops.aten.tril.default,):
        return {
            "arange": torch.ops.aten.arange.default,
            "view": torch.ops.aten.reshape.default,
            "add_scalar": torch.ops.aten.add.Scalar,
            "ge_tensor": torch.ops.aten.ge.Tensor,
            "where": torch.ops.aten.where.self,
            "scalar_tensor": torch.ops.aten.scalar_tensor.default,
            "bitwise_and": torch.ops.aten.bitwise_and.Tensor,
        }

    raise RuntimeError(f"Unable to get decomposition ops for {op}")


class DecomposeTrilPass(ArmPass):
    """
    mask_bool = (row + diagonal) >= col     (intended AOT-constant)
    out = where(mask_bool, x, 0)            (0 is a scalar tensor, broadcasted)
    """

    _passes_required_after: Set[Type[ExportPass]] = {ComputeConstantOpsAOTPass}

    def call_operator(self, op, args, kwargs, meta):
        handled_ops = [torch.ops.aten.tril.default]

        if op not in handled_ops:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]

        input_shape = list(x.data.size())
        if len(input_shape) < 2:
            raise RuntimeError(
                f"tril expects input with rank >= 2; got shape {tuple(input_shape)}"
            )

        rows = int(input_shape[-2])
        cols = int(input_shape[-1])

        ops = _get_ops(op)

        view_op = ops["view"]

        # row: (rows, 1)
        row_1d = super().call_operator(
            ops["arange"], (rows,), {"dtype": torch.float32}, meta, True
        )
        row = super().call_operator(view_op, (row_1d, [rows, 1]), {}, meta, True)

        diagonal = get_node_arg(args, 1, 0)

        if diagonal != 0:
            row = super().call_operator(
                ops["add_scalar"], (row, diagonal), {}, meta, True
            )

        # col: (1, cols)
        col_1d = super().call_operator(
            ops["arange"], (cols,), {"dtype": torch.float32}, meta, True
        )
        col = super().call_operator(view_op, (col_1d, [1, cols]), {}, meta, True)

        # mask_bool: (rows, cols)
        mask_bool = super().call_operator(ops["ge_tensor"], (row, col), {}, meta, True)

        # bool input: y = x & mask
        if x.data.dtype == torch.bool:
            return super().call_operator(
                ops["bitwise_and"], (x, mask_bool), {}, meta, True
            )

        zero = super().call_operator(
            ops["scalar_tensor"], (0,), {"dtype": x.data.dtype}, meta, True
        )

        # y = where(mask, x, 0)
        out = super().call_operator(ops["where"], (mask_bool, x, zero), {}, meta, True)
        return out
