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
    torch.ops.aten.add_.Tensor: torch.ops.aten.add.Tensor,
    torch.ops.aten.sub_.Tensor: torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul_.Tensor: torch.ops.aten.mul.Tensor,
    torch.ops.aten.div_.Tensor: torch.ops.aten.div.Tensor,
    exir_ops.edge.aten.add_.Tensor: exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub_.Tensor: exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul_.Tensor: exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div_.Tensor: exir_ops.edge.aten.div.Tensor,
}


class RewriteInplaceArithmeticPass(ArmPass):
    """Rewrite inplace arithmetic ops into functional equivalents."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        target_op = OP_MAP.get(op)
        if target_op is None:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(target_op, args, kwargs, meta, updated=True)
