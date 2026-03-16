# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


def _get_where_scalar_other_decomposition(op):
    """Return the operator overloads used to decompose where.ScalarOther to
    where.self.

    Raises:
        RuntimeError: If the provided operator is not supported by this pass.

    """
    if op is exir_ops.edge.aten.where.ScalarOther:
        return (
            exir_ops.edge.aten.full_like.default,
            exir_ops.edge.aten.where.self,
        )

    raise RuntimeError(f"Can't get where.ScalarOther decomposition for op {op}")


class DecomposeWhereScalarOtherPass(ArmPass):
    """Decompose where.ScalarOther into where.self with a tensorized scalar."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    _TARGET_OPS = {
        exir_ops.edge.aten.where.ScalarOther,
    }

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if (
            op not in DecomposeWhereScalarOtherPass._TARGET_OPS
            or not self.allowed_to_transform(meta)
        ):
            return super().call_operator(op, args, kwargs, meta, updated)

        condition, self_tensor, other_scalar = args

        full_like_op, where_op = _get_where_scalar_other_decomposition(op)

        other_tensor = super().call_operator(
            full_like_op,
            args=(self_tensor, other_scalar),
            kwargs={},
            meta=meta,
            updated=True,
        )

        return super().call_operator(
            where_op,
            (condition, self_tensor, other_tensor),
            {},
            meta,
            updated=True,
        )
