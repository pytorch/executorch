# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, List, Tuple

import torch
from executorch.exir.dialects._ops import bind_pattern_to_op, ops as exir_ops
from executorch.exir.pass_base import ExportPass

from executorch.exir.passes.replace_aten_with_edge_pass import (
    aten_to_edge,
    should_lower_to_edge,
)
from torch import fx
from torch.fx import GraphModule, subgraph_rewriter
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils import _pytree as pytree

from torch.library import impl, Library

custom_ops_lib = Library("ggml", "DEF")

custom_ops_lib.define(
    "mul_mat.out(Tensor input, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"
)

custom_ops_lib.define("mul_mat(Tensor input, Tensor mat2) -> Tensor")


def out_kernel(a, b, *, out):
    d = torch.ops.aten.view_copy.default(b, [1, 64])
    e = torch.ops.aten.mm.out(d, a, out=out)
    return out


custom_ops_lib.impl("mul_mat.out", out_kernel)


def _trace_and_lower_to_edge_ops(f: Callable) -> fx.GraphModule:
    gm = fx.symbolic_trace(f)
    for node in gm.graph.nodes:
        if node.op == "call_function" and should_lower_to_edge(node.target):
            node.target = aten_to_edge(node.target)
    gm.recompile()
    return gm


# Fuse the following pattern:
#   - d = view_copy(b, [1, 64])
#   - e = mm(d, a)


def get_patterns_and_replacements() -> List[Tuple[Callable, Callable, List[Callable]]]:
    @bind_pattern_to_op(custom_ops_lib, "mul_mat")
    def pattern(a, b):
        d = torch.ops.aten.view_copy.default(b, [1, 64])
        e = torch.ops.aten.mm.default(d, a)
        return e

    def replacement(a, b):
        return torch.ops.ggml.mul_mat.default(a, b)

    p_graph = _trace_and_lower_to_edge_ops(pattern)
    r_graph = _trace_and_lower_to_edge_ops(replacement)
    # print(p_graph.graph)
    # print(r_graph.graph)
    return [
        (
            p_graph,
            r_graph,
            [],
        )
    ]


class PermuteMMFusionPass(ExportPass):
    def __init__(self, _fix_node_meta_val=False):
        super().__init__()
        self._fix_node_meta_val = _fix_node_meta_val

    def call(self, graph_module: GraphModule) -> PassResult:
        for (
            pattern,
            replacement,
            match_filters,
        ) in get_patterns_and_replacements():
            subgraph_rewriter.replace_pattern_with_filters(
                graph_module, pattern, replacement, match_filters
            )

        if self._fix_node_meta_val:
            for n in graph_module.graph.nodes:
                if n.op == "call_function" and "val" not in n.meta:
                    args, kwargs = pytree.tree_map_only(
                        torch.fx.Node, lambda x: x.meta["val"], (n.args, n.kwargs)
                    )
                    n.meta["val"] = n.target(*args, **kwargs)
        graph_module.graph.lint()
        graph_module.graph.eliminate_dead_code()
        return PassResult(graph_module, True)
