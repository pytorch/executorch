# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator

import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ReplaceSymSumPass(PassBase):
    """
    Replaces torch.sym_sum([a, b, ...]) with a chain of operator.add.

    sym_sum takes a list of SymInts, which has no ExecuTorch prim op. It is
    emitted by PyTorch's decompositions, e.g. inside the while_loop body of
    torch.export._patches.register_lstm_while_loop_decomposition.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in list(module.graph.nodes):
                if node.op != "call_function" or node.target is not torch.sym_sum:
                    continue
                args = list(node.args[0])
                with module.graph.inserting_before(node):
                    acc = args[0]
                    for arg in args[1:]:
                        acc = module.graph.call_function(operator.add, (acc, arg))
                        acc.meta = node.meta.copy()
                node.replace_all_uses_with(acc)
                module.graph.erase_node(node)
                modified = True
            if modified:
                module.recompile()
        return PassResult(graph_module, modified)
