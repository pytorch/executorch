# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numbers
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


_ADD_OPS = (
    exir_ops.edge.aten.add.Tensor,
    torch.ops.aten.add.Tensor,
)

_SUB_OPS = (
    exir_ops.edge.aten.sub.Tensor,
    torch.ops.aten.sub.Tensor,
)


def _get_ops(op):
    if op in _ADD_OPS:
        if op is exir_ops.edge.aten.add.Tensor:
            return (
                exir_ops.edge.aten.mul.Tensor,
                exir_ops.edge.aten.full.default,
                exir_ops.edge.aten.add.Tensor,
            )
        return (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.full.default,
            torch.ops.aten.add.Tensor,
        )
    if op in _SUB_OPS:
        if op is exir_ops.edge.aten.sub.Tensor:
            return (
                exir_ops.edge.aten.mul.Tensor,
                exir_ops.edge.aten.full.default,
                exir_ops.edge.aten.sub.Tensor,
            )
        return (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.full.default,
            torch.ops.aten.sub.Tensor,
        )
    raise RuntimeError(f"Unsupported operator {op}")


def _should_decompose(alpha) -> bool:
    if isinstance(alpha, numbers.Number):
        return alpha != 1
    return False


class DecomposeAddSubAlphaPass(ArmPass):
    """Rewrite add/sub with alpha into a mul followed by add/sub."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta, updated: bool | None = False):
        if op not in _ADD_OPS + _SUB_OPS:
            return super().call_operator(op, args, kwargs, meta, updated)

        alpha = kwargs.get("alpha", 1)
        if not _should_decompose(alpha):
            return super().call_operator(op, args, kwargs, meta, updated)

        mul_op, full_op, binary_op = _get_ops(op)
        lhs, rhs = args

        alpha_full = super().call_operator(
            full_op,
            ((1,), float(alpha)),
            {"device": meta["val"].device},
            meta,
            updated=True,
        )
        scaled_rhs = super().call_operator(
            mul_op,
            (rhs, alpha_full),
            {},
            meta,
            updated=True,
        )
        return super().call_operator(
            binary_op,
            (lhs, scaled_rhs),
            {},
            meta,
            updated=True,
        )
