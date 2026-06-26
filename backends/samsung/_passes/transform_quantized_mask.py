# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.export import ExportedProgram
from torch.fx import GraphModule


class TransformQuantizedMaskPass(ExportPass):
    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def get_mask_mul(self, graph_module: GraphModule):
        """
        Iterator for each patterns in the graph.
        The obj returned by iterator is the first node of the pattern.
        """
        nodes_in_pattern = (
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten._to_copy.default,
            exir_ops.edge.aten.unsqueeze_copy.default,
            exir_ops.edge.aten.mul.Tensor,
        )
        mask_node = None
        for node in graph_module.graph.nodes:
            if node.target != "attention_mask":
                continue
            else:
                mask_node = node
                break
        if mask_node is None:
            return None
        while node.target != exir_ops.edge.aten.mul.Tensor:
            find_next = False
            for successor in list(node.users.keys()):
                if successor.target in nodes_in_pattern:
                    node = successor
                    find_next = True
                    break
            if not find_next:
                return None
        return node

    def transform(
        self,
        graph_module: GraphModule,
    ):
        mask_mul = self.get_mask_mul(graph_module)
        if mask_mul is None:
            return
        rsub_node = mask_mul.args[0]
        manual_mul_idx = 0
        for add in list(mask_mul.users.keys()):
            custom_tensor_name = f"_custom_tensor_{manual_mul_idx}"
            div_node = add.args[0]
            if "quantize_attrs" not in div_node.meta:
                return
            div_quant_args = div_node.meta["quantize_attrs"]
            custom_tensor = torch.tensor(
                (
                    div_node.meta["quantize_attrs"][QuantConstants.QUANT_KEY.quant_min]
                    - div_node.meta["quantize_attrs"][
                        QuantConstants.QUANT_KEY.zero_point
                    ]
                )
                * div_node.meta["quantize_attrs"][QuantConstants.QUANT_KEY.scale],
                dtype=torch.float32,
            )
            graph_module.register_buffer(custom_tensor_name, custom_tensor)
            add.meta["quantize_attrs"] = div_quant_args
            with graph_module.graph.inserting_after(rsub_node):
                custom_attr = graph_module.graph.get_attr(custom_tensor_name)
            with graph_module.graph.inserting_after(custom_attr):
                new_mul = graph_module.graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.mul.Tensor,
                    (mask_mul.args[0], custom_attr),
                )
                new_mul.meta["quantize_attrs"] = div_quant_args
                add.replace_input_with(mask_mul, new_mul)

            rsub_in = rsub_node.args[1]
            with graph_module.graph.inserting_before(add):
                new_mul = graph_module.graph.create_node(
                    "call_function", exir_ops.edge.aten.mul.Tensor, (div_node, rsub_in)
                )
                new_mul.meta["quantize_attrs"] = div_quant_args
                add.replace_input_with(div_node, new_mul)
            manual_mul_idx += 1

    def call(self, graph_module: GraphModule):
        self.transform(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
