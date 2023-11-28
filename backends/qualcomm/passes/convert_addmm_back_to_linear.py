# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import Counter
from typing import List

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload as edge_op
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)


class ConvertAddmmmmWithLinear(ExportPass):
    q_ops = {
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    }
    dq_ops = {
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    }

    view_copy = exir_ops.edge.aten.view_copy.default
    permute_copy = exir_ops.edge.aten.permute_copy.default
    linear = exir_ops.edge.aten.linear.default
    addmm = exir_ops.edge.aten.addmm.default
    mm = exir_ops.edge.aten.mm.default

    addmm_patterns = [
        {view_copy: 2, permute_copy: 1, addmm: 1},
        {permute_copy: 1, addmm: 1},
    ]

    def __init__(self):
        super(ConvertAddmmmmWithLinear, self).__init__()

    def _get_original_input(
        self, inputs: List[torch.fx.Node], cur_node: torch.fx.Node
    ) -> torch.fx.Node:
        while cur_node not in inputs and cur_node.args:
            cur_node = cur_node.args[0]
        return cur_node

    def _annotate_quant_attrs(
        self, gm: torch.fx.GraphModule, node: torch.fx.Node, q_node: torch.fx.Node
    ) -> torch.fx.Node:
        quant_attr_keys = [arg.name for arg in q_node.target._schema.arguments][1:]
        quant_attrs = dict.fromkeys(quant_attr_keys)

        for i in range(1, len(q_node.args)):
            attr_n = q_node.args[i]
            value = attr_n
            if type(attr_n) == torch.fx.node.Node:
                value = getattr(gm, attr_n.target)
            quant_attrs[quant_attr_keys[i - 1]] = value
        quant_attrs["encoding"] = q_node.target
        node.meta["quant_attrs"] = quant_attrs
        return node

    def _convert_addmm(self, gm: torch.fx.GraphModule, src_partition: SourcePartition):
        inputs = src_partition.input_nodes
        # output_nodes contains output node and input buffer such as argX_X
        outputs = [
            node
            for node in src_partition.output_nodes
            if node.target != torch.ops.aten.sym_size.int and node.op != "placeholder"
        ]
        assert (
            len(outputs) == 1
        ), f"Unexpected number of outputs for a torch.nn.Linear module, expecting 1 but got {outputs}"
        output = outputs[0]
        addmm_node = [n for n in src_partition.nodes if n.target == self.addmm][0]

        # weight -> permute -> input of addmm
        weight_node = addmm_node.args[2].args[0]
        input_node = addmm_node.args[1]
        bias_node = addmm_node.args[0]

        # qnn htp does not support keepdim, the view_copy(reshape) should exist for now
        if self._get_original_input(inputs, input_node).target in self.dq_ops:
            input_node = self._annotate_quant_attrs(
                gm, input_node, self._get_original_input(inputs, input_node).args[0]
            )
        args = (input_node, weight_node, bias_node)

        with gm.graph.inserting_before(output):
            linear_node = gm.graph.create_node("call_function", self.linear, args)
            linear_node.meta = addmm_node.meta
            if list(output.users)[0].target in self.q_ops:
                linear_node = self._annotate_quant_attrs(
                    gm, linear_node, list(output.users)[0]
                )
            for user in addmm_node.users.copy():
                user.replace_input_with(addmm_node, linear_node)

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(graph, [torch.nn.Linear])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                op_cnt = Counter(
                    [n.target for n in src_partition.nodes if type(n.target) == edge_op]
                )
                if op_cnt in self.addmm_patterns:
                    self._convert_addmm(graph_module, src_partition)

                # TODO Add the corresponding pattern/rewritting for mm case once we found one
                if any(n.target == self.mm for n in src_partition.nodes):
                    raise AssertionError("find a mm to linear case")

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
