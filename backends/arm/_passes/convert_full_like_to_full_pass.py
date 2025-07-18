# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class ConvertFullLikeToFullPass(ExportPass):
    """As per the full_like pytorch documentation,
    `torch.full_like(input, fill_value)` is equivalent to
    `torch.full(input.size(),
                fill_value,
                dtype=input.dtype,
                layout=input.layout,
                device=input.device
                )`
    Skip layout and device since it's not relevant for our backend.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in [
            exir_ops.edge.aten.full_like.default,
        ]:
            return super().call_operator(op, args, kwargs, meta)

        tensor = args[0].data
        full_args = (list(tensor.shape), args[1])
        full_kwargs = {"dtype": tensor.dtype}
        return super().call_operator(
            exir_ops.edge.aten.full.default, full_args, full_kwargs, meta
        )
