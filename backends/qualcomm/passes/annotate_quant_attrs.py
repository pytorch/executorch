# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
from typing import Any, Dict

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter, set_parameter
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import dq_ops, get_quant_attrs, q_ops


class AnnotateQuantAttrs(ExportPass):
    """
    Add "quant_attrs" to graph nodes' meta from the QDQ information
    generated after quatization process.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(AnnotateQuantAttrs, self).__init__()
        self.edge_program = edge_program

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

    def _expand(self, tensor, dim, axis) -> torch.Tensor:
        tensor = tensor[(...,) + (None,) * (dim - 1)]
        order = torch.arange(dim).tolist()
        order[axis], order[0] = order[0], order[axis]
        return tensor.permute(order)

    def _annotate_quant_attrs(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            # With fold_quant enabled, check if the input of dq op is quantized param.
            param = None
            if n.target in dq_ops:
                param = get_parameter(n.args[0], self.edge_program)

            if n.target not in q_ops and param is None:
                continue

            quant_attrs = get_quant_attrs(self.edge_program, n)
            self._annotate_source_nodes(n, quant_attrs)

            # We need to dequant all the fold_quant parameters.
            # If an operation is not supported by QNN and got fallback, it will expect a fp32 param.
            if param is not None:
                if quant_attrs["encoding"] in [
                    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
                ]:
                    dim, axis = param.dim(), quant_attrs["axis"]
                    scales = self._expand(quant_attrs["scales"], dim, axis)
                    offsets = self._expand(quant_attrs["zero_points"], dim, axis)
                    param = (
                        param.sub(offsets).mul(scales).to(torch.float32).contiguous()
                    )
                    set_parameter(param, n.args[0], self.edge_program)
                else:
                    scale = quant_attrs["scale"]
                    offset = quant_attrs["zero_point"]
                    param = param.sub(offset).mul(scale).to(torch.float32).contiguous()
                    set_parameter(param, n.args[0], self.edge_program)

                n.args[0].meta["val"] = param
        return graph_module

    def call(self, graph_module: torch.fx.GraphModule):
        self._annotate_quant_attrs(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
