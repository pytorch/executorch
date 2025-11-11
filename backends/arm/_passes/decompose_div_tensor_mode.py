# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.decompose_div_pass import DecomposeDivPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_div_mode_ops = (exir_ops.edge.aten.div.Tensor_mode,)
aten_div_mode_ops = (torch.ops.aten.div.Tensor_mode,)

edge_unary = {
    "div": exir_ops.edge.aten.div.Tensor,
    "floor": exir_ops.edge.aten.floor.default,
    "ceil": exir_ops.edge.aten.ceil.default,
    "full": exir_ops.edge.aten.full.default,
    "lt": exir_ops.edge.aten.lt.Tensor,
    "where": exir_ops.edge.aten.where.self,
}

aten_unary = {
    "div": torch.ops.aten.div.Tensor,
    "floor": torch.ops.aten.floor.default,
    "ceil": torch.ops.aten.ceil.default,
    "full": torch.ops.aten.full.default,
    "lt": torch.ops.aten.lt.Tensor,
    "where": torch.ops.aten.where.self,
}


def _get_opset(op):
    if op in edge_div_mode_ops:
        return edge_unary
    if op in aten_div_mode_ops:
        return aten_unary
    raise RuntimeError(f"div.Tensor_mode not supported for op {op}")


class DecomposeDivTensorModePass(ArmPass):
    """
    Rewrites aten.div.Tensor_mode into

    rounding_mode=None  -> div(a, b)
    rounding_mode='floor' -> floor(div(a, b))
    rounding_mode='trunc' -> where(div(a,b) < 0, ceil(div(a,b)), floor(div(a,b)))
    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeDivPass}

    def call_operator(self, op, args, kwargs, meta):
        if op not in (edge_div_mode_ops + aten_div_mode_ops):
            return super().call_operator(op, args, kwargs, meta)

        opset = _get_opset(op)

        a, b = args[0], args[1]
        rounding_mode = kwargs.get("rounding_mode", None)
        if rounding_mode is None and len(args) > 2:
            rounding_mode = args[2]

        q = super().call_operator(opset["div"], (a, b), {}, meta, updated=True)

        if rounding_mode is None:
            return q

        if rounding_mode == "floor":
            return super().call_operator(opset["floor"], (q,), {}, meta, updated=True)

        if rounding_mode == "trunc":
            zero = super().call_operator(
                opset["full"],
                args=((1,) * len(meta["val"].size()), 0.0),
                kwargs={"dtype": torch.float32},
                meta=meta,
                updated=True,
            )
            lt0 = super().call_operator(opset["lt"], (q, zero), {}, meta, updated=True)
            ceilq = super().call_operator(opset["ceil"], (q,), {}, meta, updated=True)
            floorq = super().call_operator(opset["floor"], (q,), {}, meta, updated=True)
            return super().call_operator(
                opset["where"], (lt0, ceilq, floorq), {}, meta, updated=True
            )

        raise RuntimeError(
            f"Unsupported rounding_mode for div.Tensor_mode: {rounding_mode!r}"
        )
