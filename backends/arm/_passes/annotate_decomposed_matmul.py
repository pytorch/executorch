# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools

from typing import List

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node

from executorch.backends.arm.tosa_quant_utils import dq_op, q_op, QuantArgs
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class AnnotateDecomposedMatmulPass(ExportPass):
    """
    torch.matmul can be decomposed in many ways, for instance:
    dq -> matmul -> q can become
    dq -> repeat -> view -> bmm -> view -> dq which makes quantization folding
    difficult. This helper function find all matmul partitions and annotate its
    matmul-op (can be mm or bmm).
    """

    def _match_partition_to_node(
        self, node: torch.fx.Node, partitioned_inputs: List[torch.fx.Node]
    ) -> torch.fx.Node:
        """
        The partition.input_nodes order is not guaranteed. Compare these
        with the matmul node inputs coming in and return the nodes
        in the correct order.
        """
        if not node or node in partitioned_inputs or node.op == "placeholder":
            return node
        else:
            return self._match_partition_to_node(
                node.all_input_nodes[0], partitioned_inputs
            )
        raise RuntimeError(f"Cannot find an input node which matches, {node}.")

    def call(self, graph_module: GraphModule) -> PassResult:
        matmul_partitions = get_source_partitions(
            graph_module.graph,
            [
                torch.matmul,
            ],
            None,
        )
        matmul_partitions = list(
            itertools.chain.from_iterable(matmul_partitions.values())
        )
        matmul_targets = {
            exir_ops.edge.aten.bmm.default,
        }
        for partition in matmul_partitions:
            quantized_input = all(
                input_node.target == dq_op for input_node in partition.input_nodes
            )
            matmul_node = [
                node for node in partition.nodes if node.target in matmul_targets
            ][0]

            if quantized_input:
                matmul_args = matmul_node.all_input_nodes
                for node in matmul_args:
                    input_node = self._match_partition_to_node(
                        node, partition.input_nodes
                    )

                    # Remove partition input dq-node
                    input_node.replace_all_uses_with(input_node.all_input_nodes[0])
                    graph_module.graph.erase_node(input_node)
                    input_node_qargs = QuantArgs.from_operator(
                        input_node.target, input_node.args
                    )

                    with graph_module.graph.inserting_before(matmul_node):
                        # Create new dq-node before matmul
                        dq_node = create_node(
                            graph=graph_module.graph,
                            op_target=dq_op,
                        )
                        dq_node.args = (node, *input_node_qargs)
                        matmul_node.replace_input_with(node, dq_node)

            partition_output = list(partition.output_nodes[0].users)[0]
            quantized_output = partition_output.target == q_op
            if quantized_output:
                output_node_qargs = QuantArgs.from_operator(
                    partition_output.target, partition_output.args
                )
                with graph_module.graph.inserting_after(matmul_node):
                    # Create q-node after matmul
                    q_node = create_node(
                        graph=graph_module.graph,
                        op_target=q_op,
                    )
                    matmul_node.replace_all_uses_with(q_node)
                    q_node.args = (matmul_node, *output_node_qargs)
                # Remove partition output q-node
                partition_output.replace_all_uses_with(
                    partition_output.all_input_nodes[0]
                )
                graph_module.graph.erase_node(partition_output)

        # retrace the graph to update the fake tensor types
        graph_module = super().call(graph_module).graph_module

        graph_module.recompile()
        return PassResult(graph_module, True)
