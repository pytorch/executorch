# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_ops = (exir_ops.edge.aten.prelu.default,)
torch_ops = (torch.ops.aten.prelu.default,)


def _get_prelu_ops(op) -> tuple:
    if op in edge_ops:
        return (
            exir_ops.edge.aten.clamp.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.view_copy.default,
        )
    if op in torch_ops:
        return (
            torch.ops.aten.clamp.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.view_copy.default,
        )
    raise RuntimeError(f"Can't get decomposition ops for op {op}")


def _weight_shape(input_rank: int, weight_shape: torch.Size) -> tuple[int, ...] | None:
    weight_dims = tuple(int(dim) for dim in weight_shape)
    if len(weight_dims) == 0 or weight_dims == (1,):
        return None
    if len(weight_dims) != 1:
        raise RuntimeError(f"Unsupported PReLU weight shape: {weight_dims}")
    if input_rank < 2:
        raise RuntimeError(
            f"Per-channel PReLU weight requires input rank >= 2, got {input_rank}"
        )
    return (1, weight_dims[0], *([1] * (input_rank - 2)))


class DecomposePReLUPass(ArmOpTargetedPass):
    """Decompose PReLU into primitive TOSA-supported operations.

    PReLU(x, weight) = max(0, x) + weight * min(0, x)

    Example:
        %op1 = clamp(x,0,None) (equivalent to max(0,x))
        %op2 = clamp(x,None,0) (equivalent to min(0,x))
        %op3 = weight
        %op4 = mul(%op3,%op2)
        %op5 = add(%op1,%op4)

    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = edge_ops + torch_ops
    check_allowed_to_transform = True

    def call_operator(self, op, args, kwargs, meta):
        if (
            op not in self.target_ops
            or not self.allowed_to_transform(meta)
            or self._is_quantized_meta(meta)
        ):
            return super().call_operator(op, args, kwargs, meta)

        x, weight = args
        clamp, mul, add, view = _get_prelu_ops(op)

        positive = super().call_operator(
            op=clamp, args=(x, 0, None), kwargs=kwargs, meta=meta, updated=True
        )
        negative = super().call_operator(
            op=clamp, args=(x, None, 0), kwargs=kwargs, meta=meta, updated=True
        )

        input_rank = len(x.data.shape)
        reshape_shape = _weight_shape(input_rank, weight.data.shape)
        if reshape_shape is not None:
            weight = super().call_operator(
                op=view,
                args=(weight, reshape_shape),
                kwargs={},
                meta=meta,
            )

        scaled_negative = super().call_operator(
            op=mul,
            args=(negative, weight),
            kwargs=kwargs,
            meta=meta,
            updated=True,
        )
        return super().call_operator(
            op=add,
            args=(positive, scaled_negative),
            kwargs=kwargs,
            meta=meta,
            updated=True,
        )
