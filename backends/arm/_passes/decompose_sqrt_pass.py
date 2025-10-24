# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Tuple, Type, Union

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_sqrt_ops = (exir_ops.edge.aten.sqrt.default,)
aten_sqrt_ops = (
    torch.ops.aten.sqrt.default,
    torch.ops.aten.sqrt_.default,
)


def get_sqrt_decomposition(op) -> Union[Tuple, torch._ops.OpOverload]:
    # TODO : "MLETORCH-863 : Replace current sqrt -> pow.Tensor_Scalar workaround with pow.Tensor_Tensor"
    if op in edge_sqrt_ops:
        return exir_ops.edge.aten.pow.Tensor_Scalar
    if op in aten_sqrt_ops:
        return torch.ops.aten.pow.Tensor_Scalar
    raise RuntimeError(f"Can't get sqrt decomposition for op {op}")


class DecomposeSqrtPass(ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = {InsertTableOpsPass}

    def call_operator(self, op, args, kwargs, meta):
        """
        Decomposes `sqrt(x)` into `pow(x, 0.5)` for backend support.
        """

        if op not in (edge_sqrt_ops + aten_sqrt_ops):
            return super().call_operator(op, args, kwargs, meta)

        pow_op = get_sqrt_decomposition(op)

        return super().call_operator(pow_op, (args[0], 0.5), {}, meta, updated=True)
