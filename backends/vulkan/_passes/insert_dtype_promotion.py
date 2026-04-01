# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Set, Union

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

OpType = Union[str, torch._ops.OpOverload, EdgeOpOverload]

# Binary ops whose first two args are tensor inputs that may need promotion
BINARY_OPS: Set[OpType] = {
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.div.Tensor_mode,
    exir_ops.edge.aten.pow.Tensor_Tensor,
    exir_ops.edge.aten.minimum.default,
    exir_ops.edge.aten.eq.Tensor,
    exir_ops.edge.aten.lt.Tensor,
    exir_ops.edge.aten.le.Tensor,
    exir_ops.edge.aten.gt.Tensor,
    exir_ops.edge.aten.ge.Tensor,
}


def _promote_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    """Promote to common dtype following PyTorch type promotion rules."""
    if a == b:
        return a
    # Any mix of different dtypes promotes to float32
    return torch.float32


class InsertDtypePromotionPass(ExportPass):
    """
    Insert _to_copy nodes before binary ops when the two tensor inputs have
    different dtypes, promoting both to a common dtype.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        dirty = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in BINARY_OPS:
                continue

            lhs = node.args[0]
            rhs = node.args[1]

            if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
                continue

            if "val" not in lhs.meta or "val" not in rhs.meta:
                continue

            lhs_dtype = lhs.meta["val"].dtype
            rhs_dtype = rhs.meta["val"].dtype

            if lhs_dtype == rhs_dtype:
                continue

            promoted = _promote_dtype(lhs_dtype, rhs_dtype)

            if lhs_dtype != promoted:
                with graph_module.graph.inserting_before(node):
                    cast_lhs = graph_module.graph.create_node(
                        "call_function",
                        exir_ops.edge.aten._to_copy.default,
                        (lhs,),
                        {"dtype": promoted},
                    )
                    cast_lhs.meta["val"] = lhs.meta["val"].to(promoted)
                node.replace_input_with(lhs, cast_lhs)
                dirty = True

            if rhs_dtype != promoted:
                with graph_module.graph.inserting_before(node):
                    cast_rhs = graph_module.graph.create_node(
                        "call_function",
                        exir_ops.edge.aten._to_copy.default,
                        (rhs,),
                        {"dtype": promoted},
                    )
                    cast_rhs.meta["val"] = rhs.meta["val"].to(promoted)
                node.replace_input_with(rhs, cast_rhs)
                dirty = True

        if dirty:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            dead_code_elimination_pass(graph_module)

        return PassResult(graph_module, dirty)
