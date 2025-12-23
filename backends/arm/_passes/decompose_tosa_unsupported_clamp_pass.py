# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeTOSAUnsupportedClampPass(ArmPass):
    """Rewrite TOSA unsupported clamp into min/max chain since TOSA lacks int32 clamp support
    and only supports scalar min/max values."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    _supported_ops = {
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.aten.clamp.Tensor,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp.Tensor,
    }

    def _ensure_tensor(
        self,
        value,
        ref_tensor,
        dtype,
        rank,
        meta,
    ):
        if value is None:
            return None
        return super().call_operator(
            exir_ops.edge.aten.full.default,
            ((1,) * rank, value),
            {"dtype": dtype},
            meta,
            updated=True,
        )

    def call_operator(self, op, args, kwargs, meta):
        val = meta["val"]

        is_scalar_clamp = op in {
            exir_ops.edge.aten.clamp.default,
            torch.ops.aten.clamp.default,
        }
        is_tensor_clamp = op in {
            exir_ops.edge.aten.clamp.Tensor,
            torch.ops.aten.clamp.Tensor,
        }

        if op not in self._supported_ops:
            return super().call_operator(op, args, kwargs, meta)

        # Only rewrite scalar clamp for int32
        if is_scalar_clamp and val.dtype != torch.int32:
            return super().call_operator(op, args, kwargs, meta)

        input_tensor = args[0]
        dtype = val.dtype
        rank = len(val.shape)
        min_arg = args[1] if len(args) > 1 else None
        max_arg = args[2] if len(args) > 2 else None

        if is_scalar_clamp:
            # Scalar min/max -> make them tensors for min/max ops
            min_arg = self._ensure_tensor(min_arg, input_tensor, dtype, rank, meta)
            max_arg = self._ensure_tensor(max_arg, input_tensor, dtype, rank, meta)
        else:
            # Tensor variant: arguments are already tensors; nothing extra to do
            if not is_tensor_clamp:
                raise RuntimeError(
                    f"DecomposeTOSAUnsupportedClampPass: unexpected op {op} in tensor clamp branch"
                )

        current = input_tensor
        if min_arg is not None:
            current = super().call_operator(
                exir_ops.edge.aten.maximum.default,
                (current, min_arg),
                {},
                meta,
                updated=True,
            )
        if max_arg is not None:
            current = super().call_operator(
                exir_ops.edge.aten.minimum.default,
                (current, max_arg),
                {},
                meta,
                updated=True,
            )
        return current
