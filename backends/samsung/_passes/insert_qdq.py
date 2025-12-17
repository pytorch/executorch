# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict

import torch
from executorch.backends.samsung._passes.utils import none_quant_tensor_quant_meta
from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.backends.samsung.utils.utils import is_graph_input, is_graph_output

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export import ExportedProgram
from torch.fx import GraphModule


class QType(Enum):
    Quant = 0
    Dequant = 1


class InsertQDQPass(ExportPass):
    QDQ_MAP = {
        # per tensor
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default: exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor: exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        # per channel
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default: exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    }

    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def _create_qdq_node(
        self,
        graph_module: GraphModule,
        qtype: QType,
        input_node: torch.fx.Node,
        quant_attrs: Dict[str, Any],
    ) -> torch.fx.Node:
        assert (target := quant_attrs.get("target")), ""
        new_node_args = [input_node]
        new_node_meta_val = input_node.meta["val"]
        new_node_quant_attrs = {}
        if qtype == QType.Dequant:
            target = self.QDQ_MAP[target]
        else:
            # For input node, we should set the val type as quant type
            key = QuantConstants.QUANT_KEY.quant_dtype
            new_node_meta_val = new_node_meta_val.to(quant_attrs[key])
            new_node_quant_attrs.update(quant_attrs)

        for arg in target._schema.arguments[1:]:
            name = arg.name
            if name == "out_dtype":
                continue
            if qtype == QType.Quant:
                key = QuantConstants.QUANT_OPS_KEY_MAP[target].get(name, name)
            else:
                key = QuantConstants.DEQUANT_OPS_KEY_MAP[target].get(name, name)
            arg_value = quant_attrs[key]
            if isinstance(arg.type, torch.Tensor) and (
                isinstance(arg_value, int) or isinstance(arg_value, float)
            ):
                arg_value = torch.Tensor(arg_value)
            new_node_args.append(arg_value)

        new_node = graph_module.graph.create_node(
            "call_function", target, tuple(new_node_args)
        )
        if new_node_quant_attrs:
            new_node.meta["quantize_attrs"] = new_node_quant_attrs
        else:
            new_node.meta["quantize_attrs"] = {
                QuantConstants.QUANT_KEY.quant_dtype: torch.float32,
                QuantConstants.QUANT_KEY.scale: [1.0],
                QuantConstants.QUANT_KEY.zero_point: [0],
            }
        new_node.meta["val"] = new_node_meta_val
        return new_node

    def _add_dq_after(self, graph_module: GraphModule, node: torch.fx.Node):
        if not (quant_attrs := node.meta.get("quantize_attrs")):
            return
        with graph_module.graph.inserting_after(node):
            new_node = self._create_qdq_node(
                graph_module, QType.Dequant, node, quant_attrs
            )
            users = [user for user in node.users.keys() if (user.op == "output")]
            for user in users:
                user.replace_input_with(node, new_node)

    def _add_q_after(self, graph_module: GraphModule, node: torch.fx.Node):
        # In node don't need quant attrs after insert new quantize node.
        if not (quant_attrs := node.meta.pop("quantize_attrs", None)):
            return
        node.meta["quantize_attrs"] = none_quant_tensor_quant_meta()
        with graph_module.graph.inserting_after(node):
            users = list(node.users.keys())
            new_node = self._create_qdq_node(
                graph_module, QType.Quant, node, quant_attrs
            )
            for user in users:
                if user.target not in QuantConstants.QUANT_OPS_KEY_MAP:
                    user.replace_input_with(node, new_node)

    def _add_q_before(
        self,
        graph_module: GraphModule,
        node: torch.fx.Node,
        from_node: torch.fx.Node,
        quantize_attrs: Dict,
    ):
        with graph_module.graph.inserting_before(node):
            new_quant_node = self._create_qdq_node(
                graph_module, QType.Quant, from_node, quantize_attrs
            )
            node.replace_input_with(from_node, new_quant_node)
        return new_quant_node

    def _add_dq_before(
        self,
        graph_module: GraphModule,
        node: torch.fx.Node,
        from_node: torch.fx.Node,
        quantize_attrs: Dict,
    ):
        with graph_module.graph.inserting_before(node):
            new_dequant_node = self._create_qdq_node(
                graph_module, QType.Dequant, from_node, quantize_attrs
            )
            node.replace_input_with(from_node, new_dequant_node)
        return new_dequant_node

    def _add_qdq_for_requantize(self, graph_module: GraphModule):
        for node in graph_module.graph.nodes:
            requant_map: Dict[int, Dict] = node.meta.get("requantize")
            if requant_map is None:
                continue
            assert (ori_quant_attrs := node.meta.get("quantize_attrs"))
            usr_list = list(node.users.keys())
            for user_idx, requant_params in requant_map.items():
                user = usr_list[user_idx]
                q_node = self._add_q_before(graph_module, user, node, requant_params)
                _ = self._add_dq_before(graph_module, q_node, node, ori_quant_attrs)

    def _add_qdq(self, graph_module: GraphModule):
        for node in list(graph_module.graph.nodes):
            if is_graph_input(self.edge_program, node):
                self._add_q_after(graph_module, node)
            elif is_graph_output(node):
                self._add_dq_after(graph_module, node)

    def call(self, graph_module: GraphModule):
        self._add_qdq(graph_module)
        self._add_qdq_for_requantize(graph_module)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
