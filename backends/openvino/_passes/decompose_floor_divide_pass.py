# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

# Ops to match
DIV_TENSOR_MODE_OPS = {
    exir_ops.edge.aten.div.Tensor_mode,
    torch.ops.aten.div.Tensor_mode,
}

# Replacement op sets per dialect
EDGE_OPS = {
    "div": exir_ops.edge.aten.div.Tensor,
    "floor": exir_ops.edge.aten.floor.default,
}

ATEN_OPS = {
    "div": torch.ops.aten.div.Tensor,
    "floor": torch.ops.aten.floor.default,
}


def _get_opset(op):
    if op is exir_ops.edge.aten.div.Tensor_mode:
        return EDGE_OPS
    if op is torch.ops.aten.div.Tensor_mode:
        return ATEN_OPS
    raise RuntimeError(f"Unexpected op: {op}")


class DecomposeFloorDividePass(ExportPass):
    """Decompose div with rounding_mode='floor' for correct semantics.

    ExecuTorch decomposes floor_divide into aten.div.Tensor_mode with
    rounding_mode='floor'. OpenVINO implements this with truncation-toward-zero
    semantics instead of PyTorch's floor-toward-negative-infinity.

    This pass replaces div(x, y, rounding_mode='floor') with
    floor(div(x, y)), ensuring correct results for negative operands.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in DIV_TENSOR_MODE_OPS:
                continue

            rounding_mode = node.kwargs.get("rounding_mode")
            if rounding_mode != "floor":
                continue

            opset = _get_opset(node.target)
            a, b = node.args[0], node.args[1]

            with graph.inserting_before(node):
                div_node = graph.call_function(opset["div"], (a, b))
                result = graph.call_function(opset["floor"], (div_node,))
                node.replace_all_uses_with(result)
            graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
