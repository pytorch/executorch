# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class DecomposeSilu(ExportPass):
    def __init__(self):
        super(DecomposeSilu, self).__init__()

    def _copy_meta(self, meta: Dict):
        copied = {}
        for k, v in meta.items():
            copied[k] = v
        return copied

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.nn.functional.silu])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:

                inputs = src_partition.input_nodes
                silu_node = src_partition.nodes[0]
                with graph_module.graph.inserting_after(inputs[0]):
                    sigmoid_node = graph.create_node(
                        "call_function", torch.ops.aten.sigmoid, (inputs[0],)
                    )
                    sigmoid_node.meta = self._copy_meta(silu_node.meta)
                    with graph_module.graph.inserting_after(sigmoid_node):
                        mul_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.mul,
                            (inputs[0], sigmoid_node),
                        )
                        mul_node.meta = self._copy_meta(silu_node.meta)
                        for user in silu_node.users.copy():
                            user.replace_input_with(silu_node, mul_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
