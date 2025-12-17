# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_ops = (exir_ops.edge.aten.leaky_relu.default,)
torch_ops = (torch.ops.aten.leaky_relu.default,)


def _get_leaky_relu_ops(op) -> tuple:
    if op in edge_ops:
        return (
            exir_ops.edge.aten.clamp.default,
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.add.Tensor,
        )
    elif op in torch_ops:
        return (
            torch.ops.aten.clamp.default,
            torch.ops.aten.full.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.add.Tensor,
        )
    else:
        raise RuntimeError(f"Can't get decomposition ops for op {op}")


class DecomposeLeakyReLUPass(ArmPass):
    """
    This pass decomposes Leaky ReLU into primitive operations.
    LeakyReLU(x,slope) = max(0,x) + slope * min(0,x)

    Example:
        %op1 = clamp(x,0,None) (equivalent to max(0,x))
        %op2 = clamp(x,None,0) (equivalent to min(0,x))
        %op3 = full(x.shape,slope)
        %op4 = mul(%op3,%op2)
        %op5 = add(%op1,%op4)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_ops + torch_ops):
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        slope = args[1] if len(args) > 1 else 0.01
        dtype = x.node.meta["val"].dtype
        clamp, full, mul, add = _get_leaky_relu_ops(op)
        op1 = super().call_operator(
            op=clamp, args=(x, 0, None), kwargs=kwargs, meta=meta
        )
        op2 = super().call_operator(
            op=clamp, args=(x, None, 0), kwargs=kwargs, meta=meta
        )
        op3 = super().call_operator(
            op=full,
            args=(x.node.meta["val"].shape, slope),
            kwargs={"dtype": dtype},
            meta=meta,
        )
        op4 = super().call_operator(op=mul, args=(op3, op2), kwargs=kwargs, meta=meta)
        op5 = super().call_operator(op=add, args=(op1, op4), kwargs=kwargs, meta=meta)
        return op5
