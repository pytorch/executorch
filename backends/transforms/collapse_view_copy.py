# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class CollapseViewCopyPass(ExportPass):
    """Collapse consecutive view_copy nodes into a single view_copy.

    view_copy(view_copy(x, shape1), shape2) -> view_copy(x, shape2)

    Only the final shape matters, so intermediate view_copys can be dropped. If
    the collapsed chain reproduces the original input's shape it is an identity
    and removed entirely.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        view_copy_target = exir_ops.edge.aten.view_copy.default

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != view_copy_target:
                continue

            parent = node.args[0]
            if not (
                isinstance(parent, Node)
                and parent.op == "call_function"
                and parent.target == view_copy_target
                and len(parent.users) == 1
            ):
                continue

            original_input = parent.args[0]
            target_shape = node.args[1]

            # Compare meta shapes (not args) so SymInt dims are handled. Guard
            # with try/except because shapes may hold unbacked SymInts (e.g.
            # from .item()) that cannot be compared.
            original_val = (
                original_input.meta.get("val")
                if isinstance(original_input, Node)
                else None
            )
            output_val = node.meta.get("val")
            is_identity = False
            if original_val is not None and output_val is not None:
                try:
                    is_identity = original_val.shape == output_val.shape
                except Exception:
                    is_identity = False

            if is_identity:
                node.replace_all_uses_with(original_input)
                graph.erase_node(node)
                graph.erase_node(parent)
            else:
                node.args = (original_input, target_shape)
                graph.erase_node(parent)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()

        return PassResult(graph_module, modified)
