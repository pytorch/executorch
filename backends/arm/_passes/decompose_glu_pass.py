# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


# For FP case
edge_glu = exir_ops.edge.aten.glu.default

# For INT case
aten_glu = torch.ops.aten.glu.default


def get_ops(op):
    """Returns the appropriate operator functions based on the input operator."""
    if op == edge_glu:
        return (
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sigmoid.default,
            exir_ops.edge.aten.slice_copy.Tensor,
        )
    elif op == aten_glu:
        return (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.slice_copy.Tensor,
        )
    else:
        raise ValueError(f"Unsupported operator: {op}")


class DecomposeGluPass(ArmPass):
    """Decomposes the GLU operator into hadamard product and sigmoid."""

    _passes_required_after: Set[Type[ExportPass]] = {InsertTableOpsPass}

    def call_operator(self, op, args, kwargs, meta):
        if op not in [edge_glu, aten_glu]:
            return super().call_operator(op, args, kwargs, meta)

        hadamard_prod, sigmoid, slice_op = get_ops(op)
        X = args[0]

        dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)

        if "val" not in X.node.meta:
            raise Exception("Could not get dimension metadata in input.")

        if dim < 0:
            dim += X.node.meta["val"].dim()

        n = X.node.meta["val"].size(dim)

        if n % 2:
            raise RuntimeError(
                f"glu expects an even split along dim={dim}, got size {n}"
            )

        middle = n // 2

        T1 = super().call_operator(
            slice_op, (X, dim, 0, middle), {}, meta, updated=True
        )

        T2 = super().call_operator(
            slice_op, (X, dim, middle, n), {}, meta, updated=True
        )

        T2_sigmoid = super().call_operator(sigmoid, (T2,), {}, meta, updated=True)

        return super().call_operator(
            hadamard_prod, (T1, T2_sigmoid), {}, meta, updated=True
        )
