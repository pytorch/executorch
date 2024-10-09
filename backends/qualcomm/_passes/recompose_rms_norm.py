# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .utils import dq_ops


class RecomposeRmsNorm(ExportPass):
    """
    Merge decomposed operators back to one super node.
    """

    def __init__(self):
        super().__init__()

    def _get_eps_node(self, nodes):
        # eps: one of inputs of add node
        add_node = [n for n in nodes if hasattr(n, "name") and "add" in n.name][0]
        for a in add_node.args:
            if isinstance(a, float) or a.op != "call_function":
                return a

    def _get_gamma_node(self, output_node):
        # gamma: one of inputs of output node
        for a in output_node.args:
            if a.op != "call_function" or a.target in dq_ops:
                return a

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.nn.RMSNorm])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                input_len = len(src_partition.input_nodes)
                if input_len == 1:
                    input_node = src_partition.input_nodes[0]
                elif input_len == 2:
                    inp_0, inp_1 = src_partition.input_nodes
                    input_node = inp_0 if len(inp_0.users) == 2 else inp_1
                else:
                    raise RuntimeError(
                        f"Found a edge case of rms_node partitoin {src_partition}, which has {input_len} inputs"
                    )

                output_node = src_partition.output_nodes[0]
                eps_node = self._get_eps_node(src_partition.nodes)
                gamma_node = self._get_gamma_node(output_node)

                with graph.inserting_before(output_node):
                    # args schema
                    # (Tensor input, int[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
                    rms_node = graph.create_node(
                        "call_function",
                        exir_ops.edge.aten.rms_norm.default,
                        (
                            input_node,
                            list(gamma_node.meta["val"].shape),
                            gamma_node,
                            eps_node,
                        ),
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, rms_node)
                    # copy metadata
                    rms_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
