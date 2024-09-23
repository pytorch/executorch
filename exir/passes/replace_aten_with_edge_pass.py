# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.executorch_prim_ops_registry import _EXECUTORCH_SYM_OPS
from torch.fx.node import Target


DISALLOW_LIST = [
    torch.ops.aten._assert_scalar.default,
    torch.ops.aten._assert_async.msg,
    torch.ops.aten.scalar_tensor.default,
]


def aten_to_edge(aten_op: torch._ops.OpOverload) -> EdgeOpOverload:
    # Assume qualified op name: aten::add.Tensor
    op_namespace, op_name, op_overload_name = (
        aten_op.namespace,
        aten_op._schema.name.split("::")[1],
        aten_op._overloadname,
    )
    edge_op = getattr(
        getattr(getattr(ops.edge, op_namespace), op_name), op_overload_name
    )
    return edge_op


def should_lower_to_edge(op: Target) -> bool:
    """Returns true if the given operator should be lowered to edge op."""
    return (
        isinstance(op, torch._ops.OpOverload)
        and op not in _EXECUTORCH_SYM_OPS
        and op not in DISALLOW_LIST
    )


class OpReplacePass(ExportPass):
    """
    Goes through all ops and replaces torch (aten + custom) ops with edge ops.
    Exclude those ops that don't care about input dtypes and out variants.
    """

    def __init__(self) -> None:
        super().__init__()

    def call_operator(self, op, args, kwargs, meta):
        if should_lower_to_edge(op):
            return super().call_operator(aten_to_edge(op), args, kwargs, meta)
        else:
            return super().call_operator(op, args, kwargs, meta)
