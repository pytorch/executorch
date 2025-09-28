# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


# For MI case
edge_sign = exir_ops.edge.aten.sign.default
# For BI case
aten_sign = torch.ops.aten.sign.default


def get_ops(op):
    """Returns the appropriate operator functions based on the input operator."""
    if op == edge_sign:
        return (
            exir_ops.edge.aten.gt.Scalar,
            exir_ops.edge.aten.lt.Scalar,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.neg.default,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.add.Scalar,
        )
    elif op == aten_sign:
        return (
            torch.ops.aten.gt.Scalar,
            torch.ops.aten.lt.Scalar,
            torch.ops.aten.where.self,
            torch.ops.aten.neg.default,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.add.Scalar,
        )
    else:
        raise ValueError(f"Unsupported operator: {op}")


class DecomposeSignPass(ArmPass):
    """Decomposes the sign operator into a sequence of operations that are supported by the Arm backend."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_sign, aten_sign):
            return super().call_operator(op, args, kwargs, meta)

        gt_op, lt_op, where_op, neg_op, mul_op, add_op = get_ops(op)

        x = args[0]

        gt_mask = super().call_operator(gt_op, (x, 0.0), {}, meta, updated=True)
        lt_mask = super().call_operator(lt_op, (x, 0.0), {}, meta, updated=True)

        zeros = super().call_operator(mul_op, (x, 0.0), {}, meta, updated=True)
        ones = super().call_operator(add_op, (zeros, 1.0), {}, meta, updated=True)
        neg_ones = super().call_operator(neg_op, (ones,), {}, meta, updated=True)

        negative_tensor = super().call_operator(
            where_op, (lt_mask, neg_ones, zeros), {}, meta, updated=True
        )
        positive_tensor = super().call_operator(
            where_op, (gt_mask, ones, zeros), {}, meta, updated=True
        )

        return super().call_operator(
            where_op,
            (lt_mask, negative_tensor, positive_tensor),
            {},
            meta,
            updated=True,
        )
