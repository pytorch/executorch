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

    x lies between floor(x) and ceil(x), and its distance above floor(x) says
    which one is nearer: less than 0.5 takes floor(x), more takes ceil(x), and
    exactly 0.5 is a tie that takes whichever of the two is even.

    Example:
        %dist_to_floor = sub(x, floor(x))
        %halved = mul(floor(x), 0.5)
        %floor_is_odd = eq(sub(%halved, floor(%halved)), 0.5)
        %tie_to_even = logical_and(eq(%dist_to_floor, 0.5), %floor_is_odd)
        %take_ceil = logical_or(gt(%dist_to_floor, 0.5), %tie_to_even)
        %result = where(%take_ceil, ceil(x), floor(x))

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

        sub = exir_ops.edge.aten.sub.Tensor
        mul = exir_ops.edge.aten.mul.Scalar
        floor = exir_ops.edge.aten.floor.default
        ceil = exir_ops.edge.aten.ceil.default
        eq = exir_ops.edge.aten.eq.Scalar
        gt = exir_ops.edge.aten.gt.Scalar
        logical_and = exir_ops.edge.aten.logical_and.default
        logical_or = exir_ops.edge.aten.logical_or.default
        where = exir_ops.edge.aten.where.self

        floor_x = call(floor, x)
        dist_to_floor = call(sub, x, floor_x)

        # floor_x is odd iff floor_x / 2 has a .5 fractional part
        halved = call(mul, floor_x, 0.5)
        halved_frac = call(sub, halved, call(floor, halved))
        floor_is_odd = call(eq, halved_frac, 0.5)

        tie_to_even = call(logical_and, call(eq, dist_to_floor, 0.5), floor_is_odd)
        take_ceil = call(logical_or, call(gt, dist_to_floor, 0.5), tie_to_even)
        return call(where, take_ceil, call(ceil, x), floor_x)
