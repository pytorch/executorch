# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch

from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import dq_ops, q_ops


class InsertIOQDQ(ExportPass):
    """
    For delegated QNN subgraph, no more QDQ operators will appear after
    'fold_qdq pass'.
    This pass will insert quantize nodes right after inputs, dequantize nodes
    right before outputs according to stored quantization encodings.
    """

    q_dq_map = {
        # per tensor
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor: exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        # per channel
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default: exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    }

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(InsertIOQDQ, self).__init__()
        self.edge_program = edge_program

    def _ceate_args(self, target: torch.fx.node.Target, quant_attrs: Dict):
        ret = []

        arg_schemas = list(target._schema.arguments)[1:]
        for arg_schema in arg_schemas:
            name = arg_schema.name
            # TODO: Due to the new parameter "out_dtype" in the dequantize node,
            # it could not be found in the quant_attrs of other nodes,
            # and it will cause a key error. For now, the output type
            # of our dequantize node is only float. (by default in pytorch)
            if name == "out_dtype":
                continue
            value = quant_attrs[name]
            if type(arg_schema.type) == torch.tensor and type(value) in [int, float]:
                value = torch.tensor(value)
            ret.append(value)
        return ret

    def _create_node(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.node,
        target: torch.fx.node.Target,
        quant_attrs: Dict = None,
    ) -> torch.fx.node:
        # check if there has a specified quant_attrs
        # if not, use the existent info. from current node
        if quant_attrs is None:
            quant_attrs = node.meta.get("quant_attrs")

        inserted_node = graph_module.graph.create_node(
            "call_function",
            target,
            (node, *self._ceate_args(target, quant_attrs)),
        )
        meta_val = node.meta["val"]
        if target in self.q_dq_map:
            inserted_node.meta["quant_attrs"] = node.meta.pop("quant_attrs")
            meta_val = meta_val.to(quant_attrs["dtype"])

        inserted_node.meta["val"] = meta_val
        return inserted_node

    def _insert_quant_node(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.node,
        target: torch.fx.node.Target,
        quant_attrs: Dict = None,
    ) -> torch.fx.Node:
        with graph_module.graph.inserting_after(node):
            users = list(node.users.keys())
            inserted_node = self._create_node(graph_module, node, target, quant_attrs)
            for user in users:
                # If we found mix quantization pattern and reuse the existing q_node, we skip adding a new q node.
                if user.target not in q_ops:
                    user.replace_input_with(node, inserted_node)

        return inserted_node

    def _insert_dequant_node(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.node,
        target: torch.fx.node.Target,
    ) -> None:
        with graph_module.graph.inserting_after(node):
            users = list(node.users.keys())
            inserted_node = self._create_node(graph_module, node, target)
            for user in users:
                if user.op == "output":
                    user.replace_input_with(node, inserted_node)

    # When having requantization dq/q nodes at the input,
    # such as the case: input1 -> dq_node1 -> q_node1 -> node1,
    # we should fold the dq_node1 and connect input -> q_node1 -> node1.
    def _fold_mix_quantization_dq_node(self, graph_module, input_node):
        input_users = list(input_node.users.keys())
        for input_user in input_users:
            if input_user.target not in dq_ops:
                continue
            dq_users = list(input_user.users.keys())
            for dq_user in dq_users:
                dq_user.replace_input_with(input_user, input_node)

    # When having requantization dq/q nodes at the output,
    # such as the case: node(int32) -> dq(int32) -> q(uint8) -> output(int32),
    # we should fold the q node and connect node(int32) -> dq(int32) -> output(int32).
    def _fold_mix_quantization_q_node(self, graph_module, node, users):
        for user in users:
            if user.op == "output":
                output_node = user
                break

        dq_node = node.args[0]
        for out_node in output_node.meta["val"]:
            if dq_node.meta["val"].dtype == out_node.dtype:
                user.replace_input_with(node, dq_node)

    def _insert(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            # insert q after input or fold mix_quantization dq if applicable
            if (
                n.op == "placeholder"
                and n.meta.get("quant_attrs")
                and not is_parameter(n, self.edge_program)
            ):
                self._fold_mix_quantization_dq_node(graph_module, n)
                self._insert_quant_node(
                    graph_module, n, n.meta["quant_attrs"]["encoding"]
                )

            # insert dq before output or fold mix_quantization q if applicable
            users = list(n.users.keys())
            if n.meta.get("quant_attrs") and any(user.op == "output" for user in users):
                if n.target in q_ops:
                    self._fold_mix_quantization_q_node(graph_module, n, users)
                # If q_node is fold, it will have no users,
                # so it won't insert dequant node in following function.
                self._insert_dequant_node(
                    graph_module,
                    n,
                    self.q_dq_map[n.meta["quant_attrs"]["encoding"]],
                )

    def call(self, graph_module: torch.fx.GraphModule):
        self._insert(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
