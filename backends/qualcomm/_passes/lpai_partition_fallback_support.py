# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
from typing import Any, List, Optional

import torch
from executorch.backends.qualcomm._passes.utils import (
    insert_dequant_node,
    insert_quant_node,
)
from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_dq_map, q_ops

from executorch.backends.qualcomm.builders.utils import is_graph_input, is_graph_output

from executorch.backends.qualcomm.utils.constants import (
    QCOM_BYPASS_NODE,
    QCOM_ENCODING,
    QCOM_FALLBACK_NODE,
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
)
from executorch.exir.pass_base import ExportPass, PassResult


class LpaiPartitionFallbackSupport(ExportPass):
    """
    This pass is introduced mainly to support LPAI graphs with partition.
    For these kind of graphs, qdq nodes is wrapped around the fall back node.
    This is because CPU runs in dtype=fp32 while LPAI runs in dtype=quantized(e.g., uint8)
    One of the biggest difference between LPAI and HTP is that CPU is
    more accurate in computing q/dq nodes than LPAI, so it is better to fall back them to CPU.

    Biggest challenge here is that a graph must be Pytorch compatible while we need to make the
    final graph that runs on device correctly.

    The overall flow will looking like this:
    Sample graph: X -> Y -> Z
    1. Edge Transform
       step 1: Run this pass, find nodes not supported by LPAI backend.
               In this example, let's say Y node is not supported.
       step 2: insert qdq nodes before and after unsupported node.
               Current graph: X -> q1 -> dq1 -> Y -> q2 -> dq2 -> Z
       step 3: Tagging what nodes fall back to cpu and what nodes to erase.
               This is the part that is tricky.
               X node runs on cpu(fp32) while Y node runs on LPAI(uint8),
               which means only q1(fp32->uint8) is necessary while dq1(uint8->fp32) is not.
               Same logic applies to 2nd pair of qdq, where dq2(uint8->fp32) is necessary while q2(fp32->uint8) is not.
               In this case, we need to force dq1 and q2 fall back to cpu so the quantize and dequantize happens on cpu.
               However, for dq1 and q2, we need to make it pass parititoner so we have control over the nodes instead of falling back to cpu.
               With the control, we can later drop them in qnn_preprocess.
               We can't drop it here in this pass since we need to make the graph valid until qnn_preprocess.
    2. Qnn Preprocess:
       step 1: Before qnn_preprocess_pass with 3 graphs: X -> q1, fallback_cpu_graph, dq2 -> Z
       step 2: After qnn_preprocess pass, mainly fold_qdq, also 3 graphs: X, fallback_cpu_graph, Z
    So the finalize graph is: X(in/out: uint8) -> fallback_cpu(in: fp32, out: uint8, internal looks like dq1 -> Y -> q2) -> Z(in/out: uint8)

    Note: QCOM_BYPASS_NODE tag could be removed and should still works fine since q/dq will pass LPAI validaiton.
          This flag exists so it guarantees to pass during actual partition, so if LPAI once makes validation for q/dq to fail, this pass won't break.

    LIMITATIONS:
    1. Does not support fallback nodes with weights.
    """

    def __init__(
        self,
        edge_program: Optional[torch.export.ExportedProgram] = None,
        compiler_specs: Optional[List[Any]] = None,
        skip_node_id_set: Optional[set] = None,
        skip_node_op_set: Optional[set] = None,
    ):
        super().__init__()
        self.edge_program = edge_program
        self.compiler_specs = compiler_specs
        self.skip_node_id_set = (
            skip_node_id_set if skip_node_id_set is not None else set()
        )
        self.skip_node_op_set = (
            skip_node_op_set if skip_node_op_set is not None else set()
        )

    def preserve_io_qdq(self, graph_module: torch.fx.GraphModule) -> None:
        """
        In LPAI backend v6, there is an accuracy drop for the quantize and
        dequantize operations. To address this, keep the quantize/dequantize
        operations at the model's input and output on CPU.

        input -> q1 (Fall back) -> dq1 (Bypass) -> graph -> q2 (Bypass) -> dq2 (Fall back) -> output

        q1 and dq2 fall back to CPU so the quantize/dequantize happens on CPU,
        while dq1 and q2 are bypassed in qnn_partition and folded in qnn_preprocess.

        A boundary q/dq is only inserted when the input is consumed by (or the
        output is produced by) a delegated node. Inputs/outputs that neighbor a
        CPU fall back node already exchange fp32 with CPU and need no q/dq.
        """
        for n in list(graph_module.graph.nodes):
            if (
                is_graph_input(n, self.edge_program)
                and n.meta.get(QCOM_QUANT_ATTRS)
                and QCOM_QUANTIZED_IO not in n.meta
            ):
                input_node = n
                for user in list(input_node.users):
                    q_node = insert_quant_node(
                        graph_module=graph_module,
                        input_node=n,
                        output_node=user,
                        target=n.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING],
                        pop_quant_attrs=False,
                    )
                    q_node.meta[QCOM_FALLBACK_NODE] = True
                    dq_node = insert_dequant_node(
                        graph_module=graph_module,
                        input_node=q_node,
                        output_node=user,
                        target=q_dq_map[q_node.target],
                    )
                    dq_node.meta[QCOM_BYPASS_NODE] = True
            elif (
                is_graph_output(n)
                and n.op == "call_function"
                and n.meta.get(QCOM_QUANT_ATTRS)
                and QCOM_QUANTIZED_IO not in n.meta
            ):
                output_node = n
                for getitem_node in [user for user in n.users if user.op == "output"]:
                    q_node = insert_quant_node(
                        graph_module=graph_module,
                        input_node=output_node,
                        output_node=getitem_node,
                        target=output_node.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING],
                        pop_quant_attrs=False,
                    )
                    q_node.meta[QCOM_BYPASS_NODE] = True
                    dq_node = insert_dequant_node(
                        graph_module=graph_module,
                        input_node=q_node,
                        output_node=getitem_node,
                        target=q_dq_map[q_node.target],
                    )
                    dq_node.meta[QCOM_FALLBACK_NODE] = True

    def get_unsupported_nodes(self, graph_module: torch.fx.GraphModule):
        from executorch.backends.qualcomm.partition.qnn_partitioner import (
            QnnOperatorSupport,
        )

        # TODO: Using self.edge_program is actually a little dangerous.
        # The io node might be differ. Check node_visitor_manager.py generate_node_to_external_map.
        op_validator = QnnOperatorSupport(
            edge_program=self.edge_program,
            compiler_specs=self.compiler_specs,
            skip_node_id_set=self.skip_node_id_set,
            skip_node_op_set=self.skip_node_op_set,
            is_qnn_partitioner=False,
            phase="LpaiPartitionFallbackSupport",
        )

        unsupported_nodes = []
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in q_ops or node.target in dq_ops:
                continue
            supported = op_validator.is_node_supported(None, node)

            if not supported:
                unsupported_nodes.append(node)
        return unsupported_nodes

    def insert_partition_qdq(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> None:
        node.meta[QCOM_FALLBACK_NODE] = True
        # Inserting Q1 and DQ1 node
        # Checking QCOM_QUANT_ATTRS to ensure it is quant node
        # Checking not in dq_ops so if previous node got fall back and already has qdq nodes, don't create extra pair of qdq node.
        input_nodes = [
            input_node
            for input_node in node.all_input_nodes
            if input_node.op == "call_function"
            and input_node.meta.get(QCOM_QUANT_ATTRS)
            and input_node.target not in dq_ops
        ]

        input_q_nodes = [
            insert_quant_node(
                graph_module=graph_module,
                input_node=input_node,
                output_node=node,
                target=input_node.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING],
                pop_quant_attrs=False,
            )
            for input_node in input_nodes
        ]
        for input_q_node in input_q_nodes:
            input_q_node.meta[QCOM_BYPASS_NODE] = True
        input_dq_nodes = [
            insert_dequant_node(
                graph_module=graph_module,
                input_node=input_q_node,
                output_node=node,
                target=q_dq_map[input_q_node.target],
            )
            for input_q_node in input_q_nodes
        ]
        for input_dq_node in input_dq_nodes:
            input_dq_node.meta[QCOM_FALLBACK_NODE] = True

        # Inserting Q2 and DQ2
        # For example, following, output_nodes = [quantized_getitem] and output_user_nodes = [user1, user2]
        #                 topk --> quantized_getitem --> user1
        #                      |                     |-> user2
        #                      |-> int_getitem2(int)
        output_nodes = (
            list(node.users)
            if any(user.target == operator.getitem for user in node.users)
            else [node]
        )
        output_nodes = [
            output_node
            for output_node in output_nodes
            if output_node.meta.get(QCOM_QUANT_ATTRS)
        ]
        for output_node in output_nodes:
            output_user_nodes = list(output_node.users)
            for output_user_node in output_user_nodes:
                output_q_node = insert_quant_node(
                    graph_module=graph_module,
                    input_node=output_node,
                    output_node=output_user_node,
                    target=output_node.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING],
                    pop_quant_attrs=False,
                )
                output_q_node.meta[QCOM_FALLBACK_NODE] = True
                output_dq_node = insert_dequant_node(
                    graph_module=graph_module,
                    input_node=output_q_node,
                    output_node=output_user_node,
                    target=q_dq_map[
                        output_q_node.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING]
                    ],
                )
                output_dq_node.meta[QCOM_BYPASS_NODE] = True
        graph_module.graph.eliminate_dead_code()

    def handle_back_to_back_nodes(self, graph_module: torch.fx.GraphModule):
        """
        This function takes care of following cases:
        1. When 2 contiguous fall back nodes ``a`` and ``b`` (both
           QCOM_FALLBACK_NODE), insert_partition_qdq generates the following graph:
           ... -> a -> q -> dq -> b -> ...
           The ``q``/``dq`` in middle can be folded for performance and accuracy optimization.

        2. For case like input_node -> q -> dq -> fall_back_node, q/dq node will be folded.

        3. For case like fall_back_node -> q -> dq -> output, q/dq node will be folded.
        """
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in q_ops or node.target in dq_ops:
                continue

            if node.meta.get(QCOM_FALLBACK_NODE) and QCOM_QUANT_ATTRS in node.meta:

                # Skip node like get_attr
                input_call_func_nodes = [
                    input_node
                    for input_node in node.all_input_nodes
                    if input_node.op == "call_function"
                ]
                assert all(
                    input_node.target in dq_ops for input_node in input_call_func_nodes
                ), (
                    f"Expected every call_function input of fp fall back node {node.name} to be "
                    f"a dequantize op (insert_partition_qdq should have wrapped "
                    f"each input with a Q1/DQ1 pair), got: "
                    f"{[(n.name, n.target) for n in input_call_func_nodes]}"
                )
                for dq_node in input_call_func_nodes:
                    q_node = dq_node.args[0]
                    if q_node.op == "placeholder":
                        raise RuntimeError(
                            "Fallback nodes with weights is not currently supported."
                        )
                    assert (
                        q_node.target in q_ops
                    ), f"Unexpected pattern found. Expecting quantize node, but get target: {q_node.target}"

                    prev_node = q_node.args[0]
                    if (
                        (
                            prev_node.target == operator.getitem
                            and prev_node.args[0].meta.get(QCOM_FALLBACK_NODE)
                        )
                        or prev_node.meta.get(QCOM_FALLBACK_NODE)
                        or is_graph_input(prev_node, self.edge_program)
                    ):
                        node.replace_input_with(dq_node, prev_node)

        # Remove io qdq node if node before output got fallback.
        for node in graph_module.graph.output_node().all_input_nodes:
            if node.target in dq_ops:
                q_node = node.args[0]
                source_node = q_node.args[0]
                if source_node.meta.get(QCOM_FALLBACK_NODE):
                    graph_module.graph.output_node().replace_input_with(
                        node, source_node
                    )
        graph_module.graph.eliminate_dead_code()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        if self.compiler_specs is None:
            return PassResult(graph_module, False)
        self.preserve_io_qdq(graph_module)
        unsupported_nodes = self.get_unsupported_nodes(graph_module)
        for node in unsupported_nodes:
            self.insert_partition_qdq(graph_module, node)
        self.handle_back_to_back_nodes(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, bool(unsupported_nodes))
