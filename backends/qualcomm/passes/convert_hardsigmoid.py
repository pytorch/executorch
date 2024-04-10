# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class ConvertHardsigmoid(ExportPass):
    """
    Merge decomposed operators from hardsigmoid back to few super nodes
    which will be mathematically equivalent. (Since QNN currently doesn't
    support hardsigmoid)
    """

    def __init__(self, quantization_capture=False):
        self.quantization_capture = quantization_capture
        super(ConvertHardsigmoid, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.nn.Hardsigmoid])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                if exir_ops.edge.aten.hardswish.default in [
                    node.target for node in src_partition.nodes
                ]:
                    continue
                if self.quantization_capture:
                    # only one hardsigmoid op will be seen
                    input_nodes = src_partition.input_nodes
                    output_nodes = src_partition.output_nodes
                    input_node = input_nodes[0]
                    output_node = output_nodes[0]
                else:
                    in_ops_target = exir_ops.edge.aten.add.Tensor
                    out_ops_target = exir_ops.edge.aten.div.Tensor
                    input_nodes = [
                        n for n in src_partition.nodes if n.target is in_ops_target
                    ]
                    output_nodes = [
                        n for n in src_partition.nodes if n.target is out_ops_target
                    ]
                    input_node = input_nodes[0].args[0]
                    output_node = output_nodes[-1]

                with graph.inserting_after(input_node):
                    # currently QNN does not support HardSigmoid,
                    # we have to replace it with equivalent representation
                    # replace following when op is available
                    """hardsigmoid_op = exir_ops.edge.aten.hardsigmoid.default
                    hardsigmoid_node = graph.create_node(
                        "call_function", hardsigmoid_op, tuple([input_node])
                    )
                    users = output_node.users.copy()
                    for user in users:
                        user.replace_input_with(output_node, hardsigmoid_node)
                    # copy metadata
                    hardsigmoid_node.meta = output_node.meta"""
                    # need to check if we're under quantization stage
                    hardswish_op = (
                        exir_ops.edge.aten.hardswish.default
                        if not self.quantization_capture
                        else torch.ops.aten.hardswish.default
                    )
                    hardswish_node = graph.create_node(
                        "call_function", hardswish_op, (input_node,)
                    )
                    with graph.inserting_after(hardswish_node):
                        # if op came from quantization capture, the hardswish node
                        # will be decomposed again. We'll have two div node here
                        if len(output_nodes) > 1:
                            hardswish_decomposed_div_node = output_nodes[0]
                            users = hardswish_decomposed_div_node.users.copy()
                            for user in users:
                                user.replace_input_with(
                                    hardswish_decomposed_div_node, hardswish_node
                                )
                            hardswish_node.meta = hardswish_decomposed_div_node.meta
                        else:
                            # need to check if we're under quantization stage
                            div_op = (
                                exir_ops.edge.aten.div.Tensor
                                if not self.quantization_capture
                                else torch.ops.aten.div.Tensor
                            )
                            div_node = graph.create_node(
                                "call_function",
                                div_op,
                                (hardswish_node, input_node),
                            )
                            users = output_node.users.copy()
                            for user in users:
                                user.replace_input_with(output_node, div_node)
                            # copy metadata
                            hardswish_node.meta = output_node.meta
                            div_node.meta = output_node.meta

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
