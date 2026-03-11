# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Set, Type

from executorch.backends.arm._passes import ArmPass

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, ProxyValue

from torch import SymInt


class RewriteSlicePass(ArmPass):
    """Rewrite slice operations with step of 1 to TOSA slice operators."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def _fixup_start(self, start, input_shape, dim) -> int:
        """Convert negative and out-of-bounds start indices to valid positive
        indices.
        """
        idx = start or 0
        if idx < 0:
            idx = start % input_shape[dim]
        if idx >= input_shape[dim]:
            idx = input_shape[dim] - 1
        return idx

    def call_operator(self, op, args, kwargs, meta, updated=False) -> ProxyValue:
        if op not in (exir_ops.edge.aten.slice_copy.Tensor,):
            return super().call_operator(op, args, kwargs, meta, updated)

        if len(args) == 5 and args[4] != 1:
            raise ValueError(
                f"Only slice with 4 arguments and step of 1 is supported, got {len(args)} arguments and step {args[4]}"
            )
        input, dim, start, end = args[:4]
        input_shape = input.data.shape
        start_index = self._fixup_start(start, input_shape, dim)

        start_list = [0] * len(input_shape)
        start_list[dim] = start_index
        size_list = list(meta.data["val"].shape)

        if any(isinstance(dim, SymInt) for dim in start_list):
            starts = start_list
        else:
            starts = super().call_shape_operator(  # type: ignore[assignment]
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (start_list,),
                {},
                meta,
                True,
            )
        if any(isinstance(dim, SymInt) for dim in size_list):
            sizes = size_list
        else:
            sizes = super().call_shape_operator(  # type: ignore[assignment]
                exir_ops.backend.tosa.CONST_SHAPE.default,
                (size_list,),
                {},
                meta,
                True,
            )

        return super().call_operator(
            exir_ops.backend.tosa.SLICE.default,
            (input, starts, sizes),
            kwargs,
            meta,
            updated=True,
        )
