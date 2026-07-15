# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops


class SymbolicToTosaShapesPass(ArmPass):

    _passes_required_after = set()

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
        if op == torch.ops.aten.sym_size.int:
            return super().call_shape_operator(
                exir_ops.backend.tosa.DIM.default,
                (args[0],),
                {"axis": args[1]},
                meta,
            )
        return super().call_operator(op, args, kwargs, meta, updated)
