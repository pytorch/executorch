# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Dict, Tuple

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeSqrtPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()

        # We cache constant tensor for the exponent
        self._half_cache: Dict[Tuple[Any, Any], Any] = {}
        self.SQRT_TO_POW = {
            exir_ops.edge.aten.sqrt.default: exir_ops.edge.aten.pow.Tensor_Tensor,
            torch.ops.aten.sqrt.default: torch.ops.aten.pow.Tensor_Tensor,
            torch.ops.aten.sqrt_.default: torch.ops.aten.pow.Tensor_Tensor,
        }

    def _get_half_tensor(
        self,
        dtype: Any,
        device: Any,
        meta: Any,
    ) -> Any:
        # Choose a floating dtype for 0.5
        if dtype in (torch.float16, torch.float32, torch.float64):
            half_dtype = dtype
        else:
            half_dtype = torch.float32

        key = (half_dtype, device)
        if key not in self._half_cache:
            half = super().call_operator(
                exir_ops.edge.aten.full.default,
                ([], 0.5),
                {"dtype": half_dtype, "device": device},
                meta,
            )
            self._half_cache[key] = half

        return self._half_cache[key]

    def call_operator(self, op: Any, args: tuple, kwargs: dict, meta: Any) -> Any:

        if op not in self.SQRT_TO_POW:
            return super().call_operator(op, args, kwargs, meta)

        if len(args) != 1:
            raise ValueError(f"Expected 1 arg to sqrt, got {len(args)}")

        x = args[0]
        pow_op = self.SQRT_TO_POW[op]

        half = self._get_half_tensor(x.data.dtype, x.data.device, meta)

        return super().call_operator(pow_op, (x, half), {}, meta)
