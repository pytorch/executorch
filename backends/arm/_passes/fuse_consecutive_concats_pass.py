# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Set, Type

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class FuseConsecutiveConcatsPass(ArmOpTargetedPass):
    """Flatten single-use concats nested on the same dimension."""

    _passes_required_after: Set[Type[ExportPass]] = set()
    target_ops = (exir_ops.edge.aten.cat.default,)

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        concat_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.cat.default, sort=False
        )
        for node in concat_nodes:
            if _try_fuse_concat(node):
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)


def _is_cat(node: Node) -> bool:
    return node.op == "call_function" and node.target == exir_ops.edge.aten.cat.default


def _cat_inputs(node: Node) -> list[Node] | None:
    if not node.args or not isinstance(node.args[0], (list, tuple)):
        return None
    inputs = list(node.args[0])
    return (
        inputs if all(isinstance(input_node, Node) for input_node in inputs) else None
    )


def _cat_dim(node: Node) -> int | None:
    if len(node.args) > 1 and isinstance(node.args[1], int):
        return node.args[1]
    dim = node.kwargs.get("dim", None)
    return dim if isinstance(dim, int) else None


def _common_qparams(nodes: list[Node]) -> tuple[bool, Any | None]:
    qparams: list[Any] = []
    for node in nodes:
        for key in ("input_qparams", "output_qparams"):
            node_qparams = node.meta.get(key)
            if isinstance(node_qparams, dict):
                qparams.extend(node_qparams.values())
    if not qparams:
        return True, None
    return all(qparam == qparams[0] for qparam in qparams), qparams[0]


def _try_fuse_concat(node: Node) -> bool:
    inputs = _cat_inputs(node)
    dim = _cat_dim(node)
    if inputs is None or dim is None:
        return False

    new_inputs: list[Node] = []
    fused_nodes = [node]
    modified = False

    for input_node in inputs:
        nested_inputs = _cat_inputs(input_node)
        if (
            _is_cat(input_node)
            and len(input_node.users) == 1
            and _cat_dim(input_node) == dim
            and nested_inputs is not None
        ):
            replacement_inputs = nested_inputs
            fused_nodes.append(input_node)
            modified = True
        else:
            replacement_inputs = [input_node]

        new_inputs.extend(replacement_inputs)

    if not modified:
        return False

    qparams_are_uniform, qparams = _common_qparams(fused_nodes)
    if not qparams_are_uniform:
        return False

    node.args = (new_inputs, *node.args[1:])
    if qparams is not None:
        node.meta["input_qparams"] = {0: qparams}
    return True
