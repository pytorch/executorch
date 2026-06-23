# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta, get_const_node


class DecomposeLogVariants(ExportPass):
    """
    Decompose log variants [log10, log2, log1p] operations using the identities:
        log10(x) = log(x) / log(10)
        log2(x)  = log(x) / log(2)
        log1p(x) = log(1 + x)
    """

    _EDGE_OPS = {
        exir_ops.edge.aten.log10.default,
        exir_ops.edge.aten.log2.default,
        exir_ops.edge.aten.log1p.default,
    }

    def __init__(self) -> None:
        super().__init__()
        self._dispatcher = {
            # Edge dialect (post-to_edge)
            exir_ops.edge.aten.log10.default: partial(self._decompose_log_n, n=10),
            exir_ops.edge.aten.log2.default: partial(self._decompose_log_n, n=2),
            exir_ops.edge.aten.log1p.default: partial(self._decompose_log_p, p=1),
            # ATen dialect (pre-to_edge)
            torch.ops.aten.log10.default: partial(self._decompose_log_n, n=10),
            torch.ops.aten.log2.default: partial(self._decompose_log_n, n=2),
            torch.ops.aten.log1p.default: partial(self._decompose_log_p, p=1),
        }

    def _decompose_log_n(self, node, graph, graph_module, const_cache, n):
        input_node = node.args[0]
        is_edge = node.target in self._EDGE_OPS

        if is_edge:
            log_op = exir_ops.edge.aten.log.default
            div_op = exir_ops.edge.aten.div.Tensor
            attr_name = f"_log_base_{n}_constant"
            if attr_name not in const_cache:
                const_cache[attr_name] = get_const_node(
                    graph, graph_module, attr_name, math.log(n), node
                )
            div_arg = const_cache[attr_name]
        else:
            log_op = torch.ops.aten.log.default
            div_op = torch.ops.aten.div.Tensor
            div_arg = math.log(n)

        with graph.inserting_after(input_node):
            log_node = graph.create_node("call_function", log_op, (input_node,))
            log_node.meta = copy_meta(node.meta)

            with graph.inserting_after(log_node):
                div_node = graph.create_node(
                    "call_function", div_op, (log_node, div_arg)
                )
                div_node.meta = copy_meta(node.meta)

        for user in node.users.copy():
            user.replace_input_with(node, div_node)

    def _decompose_log_p(self, node, graph, graph_module, const_cache, p):
        input_node = node.args[0]
        is_edge = node.target in self._EDGE_OPS

        if is_edge:
            add_op = exir_ops.edge.aten.add.Tensor
            log_op = exir_ops.edge.aten.log.default
            attr_name = f"_log1p_addend_{p}_constant"
            if attr_name not in const_cache:
                const_cache[attr_name] = get_const_node(
                    graph, graph_module, attr_name, p, node
                )
            add_arg = const_cache[attr_name]
        else:
            add_op = torch.ops.aten.add.Tensor
            log_op = torch.ops.aten.log.default
            add_arg = p

        with graph.inserting_after(input_node):
            add_node = graph.create_node("call_function", add_op, (input_node, add_arg))
            add_node.meta = copy_meta(node.meta)

            with graph.inserting_after(add_node):
                log_node = graph.create_node("call_function", log_op, (add_node,))
                log_node.meta = copy_meta(node.meta)

        for user in node.users.copy():
            user.replace_input_with(node, log_node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        const_cache = {}

        for node in list(graph.nodes):
            if node.target in self._dispatcher:
                self._dispatcher[node.target](node, graph, graph_module, const_cache)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
