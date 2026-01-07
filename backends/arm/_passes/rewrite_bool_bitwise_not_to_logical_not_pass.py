# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class RewriteBoolBitwiseNotToLogicalNotPass(ArmPass):
    """
    Rewrites ``aten.bitwise_not`` on boolean tensors to ``aten.logical_not``.

    TOSA ``bitwise_not`` does not support boolean inputs. On boolean tensors,
    ``bitwise_not`` is equivalent to ``logical_not``, so this rewrite preserves
    semantics while enabling lowering.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _TARGET_OPS = {
        exir_ops.edge.aten.bitwise_not.default,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGET_OPS:
            return super().call_operator(op, args, kwargs, meta)

        if meta["val"].dtype == torch.bool:
            x = args[0]
            return super().call_operator(
                exir_ops.edge.aten.logical_not.default,
                (x,),
                {},
                meta,
            )

        return super().call_operator(op, args, kwargs, meta)
