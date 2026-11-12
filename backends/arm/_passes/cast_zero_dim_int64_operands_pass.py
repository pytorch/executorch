# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa.specification import context_supports_int64
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

# Operators that compute a zero-dimensional int64 operand in the dtype of a
# narrower integer tensor. Arithmetic needs int32 in TOSA.
ZERO_DIM_INT32_OPS = {
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.maximum.default,
    exir_ops.edge.aten.minimum.default,
}
ZERO_DIM_NARROW_OPS = {
    exir_ops.edge.aten.bitwise_and.Tensor,
    exir_ops.edge.aten.bitwise_or.Tensor,
    exir_ops.edge.aten.bitwise_xor.Tensor,
    exir_ops.edge.aten.where.self,
}


class CastZeroDimInt64OperandsPass(ArmPass):
    """Cast zero-dimensional int64 operands to the int8, int16 or int32 dtype
    the node computes in, as in PyTorch type promotion.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _cast_op = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
    targeted_ops = ZERO_DIM_INT32_OPS | ZERO_DIM_NARROW_OPS

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False

        if not context_supports_int64():
            return PassResult(graph_module, modified)

        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue
            output = node.meta.get("val")
            if not isinstance(output, torch.Tensor) or output.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
            ):
                continue
            for arg in node.all_input_nodes:
                value = arg.meta.get("val")
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.int64
                    or value.dim() != 0
                ):
                    continue
                with graph.inserting_before(node):
                    cast = create_node(
                        graph,
                        self._cast_op,
                        args=(arg,),
                        kwargs={"dtype": output.dtype},
                    )
                cast.meta["val"] = value.to(output.dtype)
                node.replace_input_with(arg, cast)
                modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
