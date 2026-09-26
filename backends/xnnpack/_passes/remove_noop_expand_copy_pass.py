# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemoveNoopExpandCopyPass(ExportPass):
    """
    Remove ``expand_copy`` nodes that do not change tensor shape or dtype.

    In static XNNPACK export flows, shape-specialization can turn an expand into
    a materialized copy whose input and output metadata are identical. Such a
    node is an identity for the lowered graph and can be bypassed. The pass
    leaves nodes in place whenever the output shape differs from the input
    shape.
    """

    def _is_noop_expand_copy(self, node: torch.fx.Node) -> bool:
        # TODO: Investigate moving this to a shared backend transform. Other
        # backends already carry equivalent no-op expand handling.
        if node.target != exir_ops.edge.aten.expand_copy.default:
            return False

        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            return False

        input_value = input_node.meta.get("val")
        output_value = node.meta.get("val")
        if input_value is None or output_value is None:
            return False

        return (
            input_value.dtype == output_value.dtype
            and input_value.shape == output_value.shape
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph

        for node in list(graph.nodes):
            if not self._is_noop_expand_copy(node):
                continue

            node.replace_all_uses_with(node.args[0])

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
