# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeProdPass(ArmOpTargetedPass):
    """Normalize prod.dim_int to keepdim=True plus view_copy."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (exir_ops.edge.aten.prod.dim_int,)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        match len(args):
            case 3:
                input_node, dim, keepdim = args
            case 2:
                input_node, dim = args
                keepdim = False
            case _:
                raise RuntimeError(f"Unexpected arg size {len(args)} in {op}")

        input_node = super().call_operator(
            op,
            (input_node, dim, True),
            kwargs,
            meta,
            updated=True,
        )

        if keepdim:
            return input_node

        shape = list(meta["val"].size())
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (input_node, shape),
            {},
            meta,
            updated=True,
        )
