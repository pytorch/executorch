# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeRoundPass(ArmOpTargetedPass):
    """Decomposes round(x) into round-half-to-even, matching the semantics of
    aten.round / torch.round.

    Non-tie inputs round to the nearest integer via floor(x + 0.5). Exact
    ties (x + 0.5 is integral) round to the nearest even integer: that is
    floor(x + 0.5) when it is even, or floor(x + 0.5) - 1 when it is odd.

    Example:
        %rounded_up = floor(x + 0.5)
        %is_tie = eq(%rounded_up, x + 0.5)
        %is_odd = eq(frac(%rounded_up * 0.5), 0.5)
        %adjust = logical_and(%is_tie, %is_odd)
        %result = where(%adjust, %rounded_up - 1, %rounded_up)

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    target_ops = {
        exir_ops.edge.aten.round.default,
    }

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in self.target_ops or self._is_quantized_meta(meta):
            return super().call_operator(op, args, kwargs, meta, updated)
        x = args[0]

        def call(op, *op_args):
            return super(DecomposeRoundPass, self).call_operator(
                op, op_args, kwargs, meta, updated=True
            )

        add = exir_ops.edge.aten.add.Scalar
        sub_scalar = exir_ops.edge.aten.sub.Scalar
        sub_tensor = exir_ops.edge.aten.sub.Tensor
        mul = exir_ops.edge.aten.mul.Scalar
        floor = exir_ops.edge.aten.floor.default
        eq_tensor = exir_ops.edge.aten.eq.Tensor
        eq_scalar = exir_ops.edge.aten.eq.Scalar
        logical_and = exir_ops.edge.aten.logical_and.default
        where = exir_ops.edge.aten.where.self

        x_plus_half = call(add, x, 0.5)
        rounded_up = call(floor, x_plus_half)

        is_tie = call(eq_tensor, x_plus_half, rounded_up)

        # rounded_up is odd iff frac(rounded_up / 2) == 0.5
        halved = call(mul, rounded_up, 0.5)
        halved_frac = call(sub_tensor, halved, call(floor, halved))
        is_odd = call(eq_scalar, halved_frac, 0.5)

        adjust = call(logical_and, is_tie, is_odd)
        return call(where, adjust, call(sub_scalar, rounded_up, 1.0), rounded_up)
