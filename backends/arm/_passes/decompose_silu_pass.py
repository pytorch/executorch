# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.exir.pass_base import ExportPass

aten_silu_ops = (torch.ops.aten.silu.default, torch.ops.aten.silu_.default)


class DecomposeSiluPass(ArmPass):
    """
    This pass decomposes silu into a mul and a sigmoid node.

    Example:
        y = silu(a)
    Becomes:
        x = sigmoid(a)
        y = mul(a,x)
    """

    _passes_required_after: Set[Type[ExportPass]] = {InsertTableOpsPass}

    def call_operator(self, op, args, kwargs, meta):
        if op not in (aten_silu_ops):
            return super().call_operator(op, args, kwargs, meta)
        sigmoid_op = torch.ops.aten.sigmoid.default
        mul_op = torch.ops.aten.mul.Tensor

        original = args[0]
        sigmoid = super().call_operator(sigmoid_op, (original,), {}, meta, updated=True)

        return super().call_operator(
            mul_op, (original, sigmoid), {}, meta, updated=True
        )
