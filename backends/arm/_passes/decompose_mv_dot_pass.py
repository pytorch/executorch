# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.pass_base import ExportPass


class DecomposeMvDotPass(ArmOpTargetedPass):
    """Rewrite vector products as matrix multiplication before quantization."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (torch.ops.aten.mv.default, torch.ops.aten.dot.default)
    check_allowed_to_transform = True

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        lhs, rhs = args
        if op == torch.ops.aten.dot.default:
            lhs = super().call_operator(
                torch.ops.aten.unsqueeze.default, (lhs, 0), {}, meta
            )
        rhs = super().call_operator(
            torch.ops.aten.unsqueeze.default, (rhs, 1), {}, meta
        )
        product = super().call_operator(
            torch.ops.aten.matmul.default, (lhs, rhs), {}, meta
        )
        product = super().call_operator(
            torch.ops.aten.squeeze.dim, (product, 1), {}, meta
        )
        if op == torch.ops.aten.dot.default:
            product = super().call_operator(
                torch.ops.aten.squeeze.dim, (product, 0), {}, meta
            )
        return product
