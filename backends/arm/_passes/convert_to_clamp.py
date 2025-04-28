# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_operators = {
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.relu.default,
}


def get_clamp_params(op, args) -> Tuple[float | None, float | None]:
    if op == exir_ops.edge.aten.hardtanh.default:
        return args[1], args[2]
    elif op == exir_ops.edge.aten.relu.default:
        return 0.0, None
    else:
        raise ValueError(f"Getting clamp parameters for op {op} is not implemented.")


class ConvertToClampPass(ExportPass):
    def call_operator(self, op, args, kwargs, meta):
        if op not in edge_operators:
            return super().call_operator(op, args, kwargs, meta)

        return super().call_operator(
            exir_ops.edge.aten.clamp.default,
            (args[0], *get_clamp_params(op, args)),
            {},
            meta,
        )
