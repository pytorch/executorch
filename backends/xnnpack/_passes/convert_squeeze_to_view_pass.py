# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import check_or_raise
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch.fx.experimental.symbolic_shapes import has_free_symbols


class ConvertSqueezeToViewPass(XNNPACKPass):
    """
    This pass is used to convert squeeze and unsqueeze nodes into view_copy.
    This allows them to be subsequentially lowered as static_reshape ops.
    """

    SUPPORTED_OPS = [
        exir_ops.edge.aten.squeeze_copy.dim,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.unsqueeze_copy.default,
    ]

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        node_list = list(graph.nodes)
        for node in node_list:
            if node.op == "call_function":
                if node.target in self.SUPPORTED_OPS:
                    out_shape = node.meta["val"].shape

                    # Replace up to one dynamic dimension with -1 (inferred dim).
                    new_shape = []
                    dynamic_dim_count = 0
                    for d in out_shape:
                        if has_free_symbols(d):
                            new_shape.append(-1)
                            dynamic_dim_count += 1
                        else:
                            new_shape.append(d)

                    # This constraint should be enforced by the partitioner.
                    check_or_raise(
                        dynamic_dim_count <= 1,
                        "XNN supports only one dynamic dimension",
                    )

                    with graph_module.graph.inserting_after(node):
                        view_node = graph_module.graph.create_node(
                            "call_function",
                            target=exir_ops.edge.aten.view_copy.default,
                            args=(node.args[0], new_shape),
                            kwargs=node.kwargs,
                        )

                        node.replace_all_uses_with(view_node)
                        graph_module.graph.erase_node(node)

        graph_module.recompile()
        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
