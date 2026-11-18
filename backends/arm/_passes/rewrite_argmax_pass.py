# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule


class RewriteArgmaxPass(ArmPass):
    """Lower ARGMAX and its INT32 narrowing users to one TOSA ARGMAX."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.target != exir_ops.edge.aten.argmax.default:
                continue
            casts = list(node.users)
            if not casts or not all(
                user.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
                and user.kwargs.get("dtype") == torch.int32
                for user in casts
            ):
                continue

            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
            keepdim = (
                node.args[2]
                if len(node.args) > 2
                else node.kwargs.get("keepdim", False)
            )
            if dim is None or keepdim:
                continue
            input_node = node.args[0]
            if dim < 0:
                dim += len(input_node.meta["val"].shape)
            with graph.inserting_before(node):
                replacement = create_node(
                    graph,
                    exir_ops.backend.tosa.ARGMAX.default,
                    (input_node, dim),
                    from_node=node,
                    inherit_qparams=True,
                )
            for cast_node in casts:
                cast_node.replace_all_uses_with(replacement)
                graph.erase_node(cast_node)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
