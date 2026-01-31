# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass, DecomposeDivPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


edge_tan_op = exir_ops.edge.aten.tan.default


class DecomposeTanPass(ArmPass):
    """
    Decomposes tan to sin/cos
    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeDivPass}

    def call_operator(self, op, args, kwargs, meta, updated=False):
        if op != edge_tan_op:
            return super().call_operator(op, args, kwargs, meta, updated)
        # Skip quantized tan - it is decomposed as one single table op
        if (
            len(meta.data.get("input_qparams", [])) > 0
            and len(meta.data.get("output_qparams", [])) > 0
        ):
            return super().call_operator(op, args, kwargs, meta, updated)
        if (
            len(meta.data.get("input_qparams", [])) > 0
            or len(meta.data.get("output_qparams", [])) > 0
        ):
            raise RuntimeError(
                "Mixed quantized and non-quantized inputs/outputs not supported."
            )

        x = args[0]

        sin_op = exir_ops.edge.aten.sin.default
        cos_op = exir_ops.edge.aten.cos.default
        div_op = exir_ops.edge.aten.div.Tensor

        sin = super().call_operator(sin_op, (x,), {}, meta, True)
        cos = super().call_operator(cos_op, (x,), {}, meta, True)

        out = super().call_operator(div_op, (sin, cos), {}, meta, True)

        return out
