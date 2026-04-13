# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

OP_MAP = {
    exir_ops.edge.aten.le.Tensor: exir_ops.edge.aten.ge.Tensor,
    exir_ops.edge.aten.lt.Tensor: exir_ops.edge.aten.gt.Tensor,
    torch.ops.aten.le.Tensor: torch.ops.aten.ge.Tensor,
    torch.ops.aten.lt.Tensor: torch.ops.aten.gt.Tensor,
}


class RewriteLeLtToGeGtPass(ArmPass):
    """Rewrite le/lt into ge/gt with swapped inputs."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = {*OP_MAP}

    def call_operator(self, op, args, kwargs, meta):
        if not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        target_op = OP_MAP.get(op)
        if target_op is None:
            return super().call_operator(op, args, kwargs, meta)

        lhs, rhs = args
        return super().call_operator(
            target_op,
            (rhs, lhs),
            kwargs,
            meta,
            updated=True,
        )
