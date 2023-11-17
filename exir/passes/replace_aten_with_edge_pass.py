# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes.executorch_prim_ops_registry import _EXECUTORCH_SYM_OPS
from torch.fx.node import Target


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
    return isinstance(op, torch._ops.OpOverload) and op not in _EXECUTORCH_SYM_OPS


class OpReplacePass(ExportPass):
    """
    Goes through all ops and replaces torch (aten + custom) ops with edge ops.
    Exclude those ops that don't care about input dtypes and out variants.
    """

    def __init__(self) -> None:
        super().__init__()

    def call_operator(self, op, args, kwargs, meta):
        if should_lower_to_edge(op):
            if op == torch.ops.higher_order.out_dtype:
                # get the underlying op:
                assert len(args) > 2, "Expected at least 3 args"
                underlying_op = args[0]
                assert isinstance(
                    underlying_op, torch._ops.OpOverload
                ), f"{underlying_op} is not an OpOverlaod"
                out_dtype = args[1]
                assert isinstance(out_dtype, torch.dtype), f"{out_dtype} is not a dtype"
                remaining_args = args[2:]
                edge_op = aten_to_edge(underlying_op)
                edge_op_with_out_dtype = edge_op.get_op_with_out_dtype(out_dtype)
                return super().call_operator(
                    edge_op_with_out_dtype, remaining_args, kwargs, meta
                )
            return super().call_operator(aten_to_edge(op), args, kwargs, meta)
        else:
            return super().call_operator(op, args, kwargs, meta)
