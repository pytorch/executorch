# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
from operator import attrgetter

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch.utils import _pytree as pytree


class FoldParameterSplits(ExportPass):
    """
    Replace splits and slices of a parameter with new parameters before annotation.

    nn.MultiheadAttention splits in_proj_weight at runtime when query and value
    differ, so each projection's weight is the output of a split rather than a
    parameter. A per-channel weight encoding on such a tensor is never folded
    into a static weight, and lowering fails. Folding gives each projection its
    own parameter.
    """

    split_ops = {
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.chunk.default,
        torch.ops.aten.unbind.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
    }

    def _fold(self, graph_module: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node) or source.op != "get_attr":
            return False
        graph = graph_module.graph
        sources = [
            n for n in graph.nodes if n.op == "get_attr" and n.target == source.target
        ]
        # Fold only a parameter that this split alone reads. Otherwise the
        # parameter stays and, under QAT, trains apart from its folded copy.
        if [user for n in sources for user in n.users] != [node]:
            return False
        # Symbolic sizes are nodes, possibly nested in a list.
        leaves, _ = pytree.tree_flatten((node.args[1:], node.kwargs))
        if any(isinstance(leaf, torch.fx.Node) for leaf in leaves):
            return False
        tensor = attrgetter(source.target)(graph_module)
        # A buffer can be mutated at runtime (e.g. a KV cache).
        if not isinstance(tensor, torch.nn.Parameter):
            return False

        with torch.no_grad():
            outputs = node.target(tensor, *node.args[1:], **node.kwargs)
        if isinstance(outputs, torch.Tensor):
            replaced = [(node, outputs)]
        else:
            replaced = [
                (user, outputs[user.args[1]])
                for user in node.users
                if user.target == operator.getitem
            ]
        for old, value in replaced:
            name = f"{source.target.replace('.', '_')}_{old.name}"
            # Under QAT the new parameter trains in place of its slice.
            graph_module.register_parameter(
                name,
                torch.nn.Parameter(
                    value.clone().contiguous(),
                    requires_grad=tensor.requires_grad,
                ),
            )
            with graph.inserting_before(old):
                get_attr = graph.get_attr(name)
            get_attr.meta = old.meta.copy()
            old.replace_all_uses_with(get_attr)
            graph.erase_node(old)

        if not isinstance(outputs, torch.Tensor):
            graph.erase_node(node)
        for n in sources:
            graph.erase_node(n)
        owner, _, attr = source.target.rpartition(".")
        delattr(graph_module.get_submodule(owner), attr)
        return True

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and node.target in self.split_ops:
                modified |= self._fold(graph_module, node)
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
