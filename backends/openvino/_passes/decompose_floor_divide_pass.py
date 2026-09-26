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
    "to_copy": exir_ops.edge.aten._to_copy.default,
}

ATEN_OPS = {
    "div": torch.ops.aten.div.Tensor,
    "floor": torch.ops.aten.floor.default,
    "to_copy": torch.ops.aten._to_copy.default,
}


def _get_opset(op):
    if op is exir_ops.edge.aten.div.Tensor_mode:
        return EDGE_OPS
    if op is torch.ops.aten.div.Tensor_mode:
        return ATEN_OPS
    raise RuntimeError(f"Unexpected op: {op}")


def _node_dtype(node):
    """Return the dtype of a graph node's output, or None if unknown."""
    if isinstance(node, torch.fx.Node):
        val = node.meta.get("val")
        if val is not None:
            return val.dtype
    return None


class DecomposeFloorDividePass(ExportPass):
    """Decompose div with rounding_mode='floor' for correct semantics.

    ExecuTorch decomposes floor_divide into aten.div.Tensor_mode with
    rounding_mode='floor'. OpenVINO implements this with truncation-toward-zero
    semantics instead of PyTorch's floor-toward-negative-infinity.

    For float inputs, replaces div(x, y, rounding_mode='floor') with
    floor(div(x, y)).

    For integer inputs, OpenVINO's integer division truncates toward zero, so
    floor(int_div(x, y)) still gives truncation semantics. Instead we cast to
    float32, divide, floor, then cast back:
        _to_copy(floor(div(_to_copy(x, float32), _to_copy(y, float32))), int_dtype)
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

            a_dtype = _node_dtype(a)
            is_integer = a_dtype is not None and not a_dtype.is_floating_point

            with graph.inserting_before(node):
                if is_integer:
                    a_f = graph.call_function(
                        opset["to_copy"], (a,), {"dtype": torch.float32}
                    )
                    b_f = graph.call_function(
                        opset["to_copy"], (b,), {"dtype": torch.float32}
                    )
                    div_node = graph.call_function(opset["div"], (a_f, b_f))
                    floored = graph.call_function(opset["floor"], (div_node,))
                    result = graph.call_function(
                        opset["to_copy"], (floored,), {"dtype": a_dtype}
                    )
                else:
                    div_node = graph.call_function(opset["div"], (a, b))
                    result = graph.call_function(opset["floor"], (div_node,))

                node.replace_all_uses_with(result)
            graph.erase_node(node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
