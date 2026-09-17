# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numbers
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule


_ADD_OPS = (
    exir_ops.edge.aten.add.Tensor,
    torch.ops.aten.add.Tensor,
)

_SUB_OPS = (
    exir_ops.edge.aten.sub.Tensor,
    torch.ops.aten.sub.Tensor,
)

_RSUB_OPS = (
    exir_ops.edge.aten.rsub.Scalar,
    torch.ops.aten.rsub.Scalar,
)


def _get_mul_op(op):
    if op is exir_ops.edge.aten.rsub.Scalar:
        return exir_ops.edge.aten.mul.Tensor
    return torch.ops.aten.mul.Tensor


def _get_ops(op):
    if op in _ADD_OPS:
        if op is exir_ops.edge.aten.add.Tensor:
            return (
                exir_ops.edge.aten.mul.Tensor,
                exir_ops.edge.aten.add.Tensor,
            )
        return (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.add.Tensor,
        )
    if op in _SUB_OPS:
        if op is exir_ops.edge.aten.sub.Tensor:
            return (
                exir_ops.edge.aten.mul.Tensor,
                exir_ops.edge.aten.sub.Tensor,
            )
        return (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.sub.Tensor,
        )
    raise RuntimeError(f"Unsupported operator {op}")


def _should_decompose(alpha) -> bool:
    if isinstance(alpha, numbers.Number):
        return alpha != 1
    return False


class DecomposeAddSubAlphaPass(ArmOpTargetedPass):
    """Rewrite add/sub/rsub with alpha into a mul followed by the binary op."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = _ADD_OPS + _SUB_OPS + _RSUB_OPS

    def should_run_pass(self, graph_module: GraphModule) -> bool:
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target in self.target_ops
                and (
                    "alpha" in node.kwargs
                    or (node.target in _RSUB_OPS and len(node.args) > 2)
                )
            ):
                return True
        return any(
            isinstance(child, GraphModule) and self.should_run_pass(child)
            for child in graph_module.children()
        )

    def call_operator(self, op, args, kwargs, meta, updated: bool | None = False):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated)

        if op in _RSUB_OPS:
            return self._rewrite_rsub(op, args, kwargs, meta, updated)

        alpha = kwargs.get("alpha", 1)
        if not _should_decompose(alpha):
            if isinstance(alpha, numbers.Number):
                # A unit alpha changes nothing, but the TOSA operators take two
                # arguments and reject it, so drop it rather than pass it on.
                kwargs = {k: v for k, v in kwargs.items() if k != "alpha"}
            return super().call_operator(op, args, kwargs, meta, updated)

        mul_op, binary_op = _get_ops(op)
        lhs, rhs = args

        scaled_rhs = super().call_operator(
            mul_op,
            (rhs, super().call_scalar(alpha, meta)),
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

    def _rewrite_rsub(self, op, args, kwargs, meta, updated):
        """Fold a non-unit alpha out of rsub, leaving a plain rsub behind.

        rsub(self, other, alpha) is other - alpha * self, so unlike add and sub
        the alpha scales the first operand. Scaling it here rather than building
        the subtraction keeps the second operand a scalar, which is what
        ScalarsToAttributePass expects when it rewrites rsub into sub.

        """
        alpha = args[2] if len(args) > 2 else 1
        if not _should_decompose(alpha):
            # Drop a unit alpha rather than leave a third argument behind.
            return super().call_operator(op, args[:2], kwargs, meta, updated)

        scaled_self = super().call_operator(
            _get_mul_op(op),
            (args[0], super().call_scalar(alpha, meta)),
            {},
            meta,
            updated=True,
        )
        return super().call_operator(
            op,
            (scaled_self, args[1]),
            {},
            meta,
            updated=True,
        )
