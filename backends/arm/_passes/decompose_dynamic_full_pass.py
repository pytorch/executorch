# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm._passes.unsqueeze_before_repeat_pass import (
    UnsqueezeBeforeRepeatPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeDynamicFullPass(ArmOpTargetedPass):
    """Rewrite dynamic-shape `full` into scalar `full` plus `repeat`."""

    _passes_required_after: Set[Type[ExportPass]] = {UnsqueezeBeforeRepeatPass}

    full_targets = {
        torch.ops.aten.full.default,
        exir_ops.edge.aten.full.default,
    }
    target_ops = full_targets
    repeat = exir_ops.edge.aten.repeat.default

    @staticmethod
    def _has_symbolic_extent(size: Any) -> bool:
        return isinstance(size, (list, tuple)) and any(
            not isinstance(dim, int) for dim in size
        )

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op not in self.full_targets:
            return super().call_operator(op, args, kwargs, meta, updated)

        size, fill_value = args[:2]
        if not self._has_symbolic_extent(size):
            return super().call_operator(op, args, kwargs, meta, updated)

        scalar_full = super().call_operator(
            op=op,
            args=((1,), fill_value),
            kwargs=kwargs,
            meta=meta,
            updated=True,
        )
        return super().call_operator(
            op=self.repeat,
            args=(scalar_full, size),
            kwargs={},
            meta=meta,
            updated=True,
        )
