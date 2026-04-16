# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class RewriteBoolBitwiseToLogicalPass(ArmPass):
    """Rewrites ``aten.bitwise_*`` on boolean tensors to ``aten.logical_*``.

    TOSA ``bitwise_*`` does not support boolean inputs. On boolean tensors,
    ``bitwise_*`` is equivalent to ``logical_*``, so this rewrite preserves
    semantics while enabling lowering.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _TARGET_TO_LOGICAL = {
        exir_ops.edge.aten.bitwise_not.default: exir_ops.edge.aten.logical_not.default,
        exir_ops.edge.aten.bitwise_and.Tensor: exir_ops.edge.aten.logical_and.default,
        exir_ops.edge.aten.bitwise_and.Scalar: exir_ops.edge.aten.logical_and.default,
        exir_ops.edge.aten.bitwise_or.Tensor: exir_ops.edge.aten.logical_or.default,
        exir_ops.edge.aten.bitwise_or.Scalar: exir_ops.edge.aten.logical_or.default,
        exir_ops.edge.aten.bitwise_xor.Tensor: exir_ops.edge.aten.logical_xor.default,
        exir_ops.edge.aten.bitwise_xor.Scalar: exir_ops.edge.aten.logical_xor.default,
    }

    targeted_ops = set(_TARGET_TO_LOGICAL.keys())

    def call_operator(self, op, args, kwargs, meta):
        if op not in self._TARGET_TO_LOGICAL:
            return super().call_operator(op, args, kwargs, meta)

        if meta["val"].dtype == torch.bool:
            return super().call_operator(
                self._TARGET_TO_LOGICAL[op],
                args,
                kwargs,
                meta,
                updated=True,
            )

        return super().call_operator(op, args, kwargs, meta)
