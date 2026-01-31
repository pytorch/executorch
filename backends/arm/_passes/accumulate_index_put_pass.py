# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_ops = (exir_ops.edge.aten.index_put.default, exir_ops.edge.aten.index_put_.default)
aten_ops = (torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default)


def get_ops(op):
    if op in edge_ops:
        return (
            exir_ops.edge.aten.index.Tensor,
            exir_ops.edge.aten.add.Tensor,
        )
    if op in aten_ops:
        return (
            torch.ops.aten.index.Tensor,
            torch.ops.aten.add.Tensor,
        )
    raise RuntimeError(f"Can't get index_put decomposition for op {op}")


class AccumulateIndexPutPass(ArmPass):
    """This pass adjusts the values arg when the accumulate arg is set to true for the index_put op"""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in (aten_ops + edge_ops) or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        source, indices, values = args[:3]
        accumulate = args[-1] if len(args) == 4 else False

        if accumulate:
            index_op, add_op = get_ops(op)
            gathered = super().call_operator(index_op, (source, indices), {}, meta)
            added_values = super().call_operator(add_op, (gathered, values), {}, meta)

            # Update args with accumulated values and remove accumulate flag
            new_args = (source, indices, added_values)
        else:
            # Keep first 3 args, remove accumulate flag if present
            new_args = (source, indices, values)

        return super().call_operator(op, new_args, kwargs, meta)
