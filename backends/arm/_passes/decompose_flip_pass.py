# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeFlipPass(ArmOpTargetedPass):
    """Decompose a multi-axis ``aten.flip`` into a chain of single-axis flips.

    TOSA ``REVERSE`` reverses a single ``axis``; a flip over N dims is the
    composition of N independent single-axis reversals. Each single-axis
    ``aten.flip`` is lowered to one ``REVERSE`` by ``FlipVisitor``. ``flip`` is a
    shared-qparam data-movement op, so the chained flips inherit the original
    node's quantization parameters via ``meta``.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = {exir_ops.edge.aten.flip.default}

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in self.target_ops or len(args[1]) == 1:
            return super().call_operator(op, args, kwargs, meta, updated)

        # Chain one single-axis flip per dim; empty dims falls through as a no-op.
        out = args[0]
        for dim in args[1]:
            out = super().call_operator(op, (out, [dim]), kwargs, meta, updated=True)
        return out
