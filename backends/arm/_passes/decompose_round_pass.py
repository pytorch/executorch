# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass
from torch._ops import OpOverload


Op = OpOverload | EdgeOpOverload


def _get_round_decomposition_ops(op) -> tuple[Op, Op, Op, Op, Op, Op, Op]:
    """
    Returns the (full_op, ge_op, add_op, sub_op, floor_op, ceil_op, where_op) for the
    given round operation. The ops depend on whether the round op is an aten or edge op.
    """
    if op == exir_ops.edge.aten.round.default:
        return (
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.ge.Tensor,
            exir_ops.edge.aten.add.Scalar,
            exir_ops.edge.aten.sub.Scalar,
            exir_ops.edge.aten.floor.default,
            exir_ops.edge.aten.ceil.default,
            exir_ops.edge.aten.where.self,
        )
    elif op == torch.ops.aten.round.default:
        return (
            torch.ops.aten.full.default,
            torch.ops.aten.ge.Tensor,
            torch.ops.aten.add.Scalar,
            torch.ops.aten.sub.Scalar,
            torch.ops.aten.floor.default,
            torch.ops.aten.ceil.default,
            torch.ops.aten.where.self,
        )
    raise RuntimeError(f"Can't get round decomposition ops for op {op}")


class DecomposeRoundPass(ArmPass):
    """
    For inputs >= 0, round(x) is equivalent to floor(x + 0.5), and for inputs < 0,
    round(x) is equivalent to ceil(x - 0.5). This pass decomposes the round operation into
    a sequence of more primitive operations.
    Example:
        %zero = full((1,), 0.0, dtype=torch.float32)
        %is_non_negative = ge(x, %zero)
        %plus_half = add(x, 0.5)
        %minus_half = sub(x, 0.5)
        %floor = floor(%plus_half)
        %ceil = ceil(%minus_half)
        %result = where(%is_non_negative, %floor, %ceil)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _TARGET_OPS = {
        exir_ops.edge.aten.round.default,
        torch.ops.aten.round.default,
    }

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in DecomposeRoundPass._TARGET_OPS or not self.allowed_to_transform(
            meta
        ):
            return super().call_operator(op, args, kwargs, meta, updated)
        x = args[0]
        input_dtype = x.node.meta["val"].dtype
        full, ge, add, sub, floor, ceil, where = _get_round_decomposition_ops(op)
        zero = super().call_operator(
            full,
            args=((1,), 0.0),
            kwargs={"dtype": input_dtype},
            meta=meta,
            updated=True,
        )
        is_non_negative = super().call_operator(
            ge, (x, zero), kwargs, meta, updated=True
        )
        plus_half = super().call_operator(add, (x, 0.5), kwargs, meta, updated=True)
        minus_half = super().call_operator(sub, (x, 0.5), kwargs, meta, updated=True)
        floor = super().call_operator(floor, (plus_half,), kwargs, meta, updated=True)
        ceil = super().call_operator(ceil, (minus_half,), kwargs, meta, updated=True)
        return super().call_operator(
            where,
            (is_non_negative, floor, ceil),
            kwargs,
            meta,
            updated=True,
        )
