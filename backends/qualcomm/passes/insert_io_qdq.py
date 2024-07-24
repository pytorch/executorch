# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torch

from executorch.backends.qualcomm.builders.utils import is_parameter
from executorch.backends.qualcomm.utils.constants import (
    QCOM_ENCODING,
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import q_ops


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
            if isinstance(arg_schema.type, torch.Tensor) and (
                isinstance(value, int) or isinstance(value, float)
            ):
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
            quant_attrs = node.meta.get(QCOM_QUANT_ATTRS)

        inserted_node = graph_module.graph.create_node(
            "call_function",
            target,
            (node, *self._ceate_args(target, quant_attrs)),
        )
        meta_val = node.meta["val"]
        if target in self.q_dq_map:
            inserted_node.meta[QCOM_QUANT_ATTRS] = node.meta.pop(QCOM_QUANT_ATTRS)
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

    def _insert(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            # do nothing when a node is expected to output a quant tensor
            if n.meta.get(QCOM_QUANTIZED_IO):
                continue

            # insert q after input or fold mix_quantization dq if applicable
            if (
                n.op == "placeholder"
                and n.meta.get(QCOM_QUANT_ATTRS)
                and not is_parameter(n, self.edge_program)
            ):
                self._insert_quant_node(
                    graph_module, n, n.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING]
                )

            # insert dq before output or fold mix_quantization q if applicable
            users = list(n.users.keys())
            if n.meta.get(QCOM_QUANT_ATTRS) and any(
                user.op == "output" for user in users
            ):
                self._insert_dequant_node(
                    graph_module,
                    n,
                    self.q_dq_map[n.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING]],
                )

    def call(self, graph_module: torch.fx.GraphModule):
        self._insert(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
