# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeBoolEqualityPass(ArmOpTargetedPass):
    """Decompose boolean equality into logical XOR followed by logical NOT."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (exir_ops.edge.aten.eq.Tensor,)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops or any(
            arg.data.dtype != torch.bool for arg in args
        ):
            return super().call_operator(op, args, kwargs, meta)

        xor = super().call_operator(
            exir_ops.edge.aten.logical_xor.default, args, {}, meta, updated=True
        )
        return super().call_operator(
            exir_ops.edge.aten.logical_not.default, (xor,), {}, meta, updated=True
        )
