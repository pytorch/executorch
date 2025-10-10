# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Set, Type

import torch
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
    "mul": exir_ops.edge.aten.mul.Tensor,
    "sub": exir_ops.edge.aten.sub.Tensor,
}

aten_unary = {
    "div": torch.ops.aten.div.Tensor,
    "floor": torch.ops.aten.floor.default,
    "ceil": torch.ops.aten.ceil.default,
    "full": torch.ops.aten.full.default,
    "lt": torch.ops.aten.lt.Tensor,
    "where": torch.ops.aten.where.self,
    "mul": torch.ops.aten.mul.Tensor,
    "sub": torch.ops.aten.sub.Tensor,
}


def _get_opset(op):
    if op in edge_div_mode_ops:
        return edge_unary
    if op in aten_div_mode_ops:
        return aten_unary
    raise RuntimeError(f"div.Tensor_mode not supported for op {op}")


class DecomposeDivTensorModePass(ExportPass):
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

        q = super().call_operator(opset["div"], (a, b), {}, meta)

        if rounding_mode is None:
            return q

        if rounding_mode == "floor":
            q_raw = q

            # trunc(q_raw) = where(q_raw < 0, ceil(q_raw), floor(q_raw))
            q_floor = super().call_operator(opset["floor"], (q_raw,), {}, meta)
            q_ceil = super().call_operator(opset["ceil"], (q_raw,), {}, meta)

            # a zero tensor with the right shape
            out_shape = (1,) * len(meta["val"].size())
            zero = super().call_operator(
                opset["full"],
                args=(out_shape, 0.0),
                kwargs={},
                meta=meta,
            )

            is_neg = super().call_operator(opset["lt"], (q_raw, zero), {}, meta)
            q_trunc = super().call_operator(
                opset["where"], (is_neg, q_ceil, q_floor), {}, meta
            )

            # r = a - q_trunc * b (true remainder under truncation)
            q_times_b = super().call_operator(opset["mul"], (q_trunc, b), {}, meta)
            r = super().call_operator(opset["sub"], (a, q_times_b), {}, meta)

            # Decide if we need to subtract 1:
            # for b > 0, adjust if r < 0; for b < 0, adjust if r > 0.
            b_pos = super().call_operator(opset["lt"], (zero, b), {}, meta)  # b > 0
            r_lt0 = super().call_operator(opset["lt"], (r, zero), {}, meta)  # r < 0
            r_gt0 = super().call_operator(opset["lt"], (zero, r), {}, meta)  # r > 0

            adjust_if = super().call_operator(
                opset["where"], (b_pos, r_lt0, r_gt0), {}, meta
            )

            one = super().call_operator(
                opset["full"],
                args=(out_shape, 1.0),
                kwargs={},
                meta=meta,
            )
            q_minus_1 = super().call_operator(opset["sub"], (q_trunc, one), {}, meta)

            return super().call_operator(
                opset["where"], (adjust_if, q_minus_1, q_trunc), {}, meta
            )

        if rounding_mode == "trunc":
            zero = super().call_operator(
                opset["full"],
                args=((1,) * len(meta["val"].size()), 0.0),
                kwargs={},
                meta=meta,
            )
            lt0 = self.call_operator(opset["lt"], (q, zero), {}, meta)
            ceilq = self.call_operator(opset["ceil"], (q,), {}, meta)
            floorq = self.call_operator(opset["floor"], (q,), {}, meta)
            return self.call_operator(opset["where"], (lt0, ceilq, floorq), {}, meta)

        raise RuntimeError(
            f"Unsupported rounding_mode for div.Tensor_mode: {rounding_mode!r}"
        )
