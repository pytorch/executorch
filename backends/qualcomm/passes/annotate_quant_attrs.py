# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
from typing import Any, Dict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def get_quant_attrs(graph_module: torch.fx.GraphModule, quant_node: torch.fx.Node):
    quant_attr_keys = [arg.name for arg in quant_node.target._schema.arguments][1:]
    quant_attrs = dict.fromkeys(quant_attr_keys)

    for i in range(1, len(quant_node.args)):
        attr_n = quant_node.args[i]
        value = attr_n
        if type(attr_n) == torch.fx.node.Node:
            value = getattr(graph_module, attr_n.target)
        quant_attrs[quant_attr_keys[i - 1]] = value

    quant_attrs["encoding"] = quant_node.target
    return quant_attrs


class AnnotateQuantAttrs(ExportPass):
    """
    Add "quant_attrs" to graph nodes' meta from the QDQ information
    generated after quatization process.
    """

    q_ops = {
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    }

    def __init__(self, annotate_only: bool = True):
        super(AnnotateQuantAttrs, self).__init__()

    def _annotate_source_nodes(
        self, quant_node: torch.fx.Node, quant_attrs: Dict[str, Any]
    ):
        if quant_node.args[0].target == operator.getitem:
            getitem_node = quant_node.args[0]
            getitem_node.meta["quant_attrs"] = quant_attrs
            source_n = getitem_node.args[0]
        else:
            source_n = quant_node.args[0]

        source_n.meta["quant_attrs"] = quant_attrs

    def _annotate_quant_attrs(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            if n.target not in self.q_ops:
                continue

            quant_attrs = get_quant_attrs(graph_module, n)
            self._annotate_source_nodes(n, quant_attrs)

        return graph_module

    def call(self, graph_module: torch.fx.GraphModule):
        self._annotate_quant_attrs(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
