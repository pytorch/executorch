# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.pass_base import ExportPass

aten_silu_ops = (torch.ops.aten.silu.default, torch.ops.aten.silu_.default)


class DecomposeSiluPass(ExportPass):
    """
    This pass decomposes silu into a mul and a sigmoid node.

    Example:
        y = silu(a)
    Becomes:
        x = sigmoid(a)
        y = mul(a,x)
    """

    def call_operator(self, op, args, kwargs, meta):
        if op not in (aten_silu_ops):
            return super().call_operator(op, args, kwargs, meta)
        sigmoid_op = torch.ops.aten.sigmoid.default
        mul_op = torch.ops.aten.mul.Tensor

        original = args[0]
        sigmoid = super().call_operator(sigmoid_op, (original,), {}, meta)

        return super().call_operator(mul_op, (original, sigmoid), {}, meta)
