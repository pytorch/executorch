# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

# Reduction ops that OpenVINO cannot run on boolean inputs.
SUM_OPS = {
    exir_ops.edge.aten.sum.dim_IntList,
    torch.ops.aten.sum.dim_IntList,
}
CUMSUM_OPS = {
    exir_ops.edge.aten.cumsum.default,
    torch.ops.aten.cumsum.default,
}

# Element-wise multiply, which is a boolean AND when both inputs are bool.
MUL_OPS = {
    exir_ops.edge.aten.mul.Tensor,
    torch.ops.aten.mul.Tensor,
}

# Ops in the edge dialect; everything else matched here is plain aten.
EDGE_SOURCE_OPS = {
    exir_ops.edge.aten.sum.dim_IntList,
    exir_ops.edge.aten.cumsum.default,
    exir_ops.edge.aten.mul.Tensor,
}

# Replacement ops per dialect.
EDGE_OPS = {
    "to_copy": exir_ops.edge.aten._to_copy.default,
    "sum": exir_ops.edge.aten.sum.dim_IntList,
    "cumsum": exir_ops.edge.aten.cumsum.default,
    "bitwise_and": exir_ops.edge.aten.bitwise_and.Tensor,
}

ATEN_OPS = {
    "to_copy": torch.ops.aten._to_copy.default,
    "sum": torch.ops.aten.sum.dim_IntList,
    "cumsum": torch.ops.aten.cumsum.default,
    "bitwise_and": torch.ops.aten.bitwise_and.Tensor,
}


def _get_opset(op):
    if op in SUM_OPS | CUMSUM_OPS | MUL_OPS:
        return EDGE_OPS if op in EDGE_SOURCE_OPS else ATEN_OPS
    raise RuntimeError(f"Unexpected op: {op}")


def _node_dtype(node):
    """Return the dtype of a graph node's output, or None if unknown."""
    if isinstance(node, torch.fx.Node):
        val = node.meta.get("val")
        if val is not None:
            return val.dtype
    return None


class DecomposeBooleanReductionPass(ExportPass):
    """Rewrite boolean ops that OpenVINO cannot consume directly.

    OpenVINO does not accept boolean input tensors for reductions, and treats
    boolean multiply differently from PyTorch. This pass rewrites:

        sum(bool_x, ...)    -> sum(to_int32(bool_x), ...)
        cumsum(bool_x, ...) -> cumsum(to_int32(bool_x), ...)
        mul(bool_a, bool_b) -> bitwise_and(bool_a, bool_b)

    bitwise_and preserves the boolean output type so downstream ops such as
    where() still see a mask.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            if node.target in SUM_OPS:
                self._rewrite_reduction(graph, node, "sum")
            elif node.target in CUMSUM_OPS:
                self._rewrite_reduction(graph, node, "cumsum")
            elif node.target in MUL_OPS:
                self._rewrite_bool_mul(graph, node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)

    def _rewrite_reduction(self, graph, node, kind):
        input_node = node.args[0] if node.args else None
        if input_node is None or _node_dtype(input_node) != torch.bool:
            return

        opset = _get_opset(node.target)
        with graph.inserting_before(node):
            input_as_int = graph.call_function(
                opset["to_copy"], (input_node,), {"dtype": torch.int32}
            )
            if kind == "sum":
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
                keepdim = (
                    node.args[2]
                    if len(node.args) > 2
                    else node.kwargs.get("keepdim", False)
                )
                dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
                kwargs = {"dtype": dtype} if dtype is not None else {}
                result = graph.call_function(
                    opset["sum"], (input_as_int, dim, keepdim), kwargs
                )
            else:  # cumsum
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
                dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
                kwargs = {"dtype": dtype} if dtype is not None else {}
                result = graph.call_function(
                    opset["cumsum"], (input_as_int, dim), kwargs
                )
            node.replace_all_uses_with(result)
        graph.erase_node(node)

    def _rewrite_bool_mul(self, graph, node):
        if len(node.args) < 2:
            return
        a_node, b_node = node.args[0], node.args[1]
        if _node_dtype(a_node) != torch.bool or _node_dtype(b_node) != torch.bool:
            return

        opset = _get_opset(node.target)
        with graph.inserting_before(node):
            result = graph.call_function(opset["bitwise_and"], (a_node, b_node))
            node.replace_all_uses_with(result)
        graph.erase_node(node)
