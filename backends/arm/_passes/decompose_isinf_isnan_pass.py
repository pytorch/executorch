# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeIsInfAndIsNanPass(ArmOpTargetedPass):
    """Decompose ``isinf`` and ``isnan`` into TOSA-supported operations."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    edge_isinf = exir_ops.edge.aten.isinf.default
    edge_isnan = exir_ops.edge.aten.isnan.default
    target_ops = (edge_isinf, edge_isnan)
    check_allowed_to_transform = True

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        (x,) = args
        abs_op = exir_ops.edge.aten.abs.default
        eq_op = exir_ops.edge.aten.eq.Tensor
        logical_not_op = exir_ops.edge.aten.logical_not.default
        full_op = exir_ops.edge.aten.full.default

        if op is self.edge_isnan:
            equal = super().call_operator(eq_op, (x, x), {}, meta, updated=True)
            return super().call_operator(
                logical_not_op, (equal,), {}, meta, updated=True
            )

        absolute = super().call_operator(abs_op, (x,), {}, meta, updated=True)
        infinity = super().call_operator(
            full_op,
            (x.data.shape, float("inf")),
            {"dtype": x.data.dtype},
            meta,
            updated=True,
        )
        return super().call_operator(
            eq_op, (absolute, infinity), {}, meta, updated=True
        )
