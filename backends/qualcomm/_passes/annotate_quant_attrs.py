# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
from typing import Any, Dict

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter, set_parameter
from executorch.backends.qualcomm.utils.constants import (
    QCOM_ENCODING,
    QCOM_QUANT_ATTRS,
    QCOM_REQUANTIZE,
    QCOM_SCALES,
    QCOM_ZERO_POINTS,
)
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
            getitem_node.meta[QCOM_QUANT_ATTRS] = quant_attrs
            source_n = getitem_node.args[0]
        else:
            source_n = quant_node.args[0]
        source_n.meta[QCOM_QUANT_ATTRS] = quant_attrs

    def _expand(self, tensor, dim, axis) -> torch.Tensor:
        tensor = tensor[(...,) + (None,) * (dim - 1)]
        order = torch.arange(dim).tolist()
        order[axis], order[0] = order[0], order[axis]
        return tensor.permute(order)

    # Find the the last dq node between regular op nodes
    # Return dq2 in example below when q1 is given as node parameter:
    # ... -> n1 -> q1 -> dq1 -> q2 -> dq2 -> n2 -> ...
    def _find_last_dq_node(self, node: torch.fx.node.Node) -> torch.fx.node.Node:
        if list(node.users)[0].target in q_ops.union(dq_ops):
            return self._find_last_dq_node(list(node.users)[0])
        return node

    def _annotate_requant(self, n):
        # Record requant attributes:
        # node1 -> q_ui8 -> dq_ui8 -> q_int32 -> dq_int32 -> node2 -> ....
        # We store quant info for dq_ui8 and q_int32 in node1.meta
        if n.target in q_ops and n.args[0].target not in dq_ops:
            dq_node = self._find_last_dq_node(n)
            q_attrs = get_quant_attrs(self.edge_program, n)
            dq_attrs = get_quant_attrs(self.edge_program, dq_node)

            # TODO: Store multiple pairs of requantize attributes when we have an op builder
            # that has multiple outputs that requires quant attributes.
            if q_attrs["dtype"] != dq_attrs["dtype"]:
                dq_attrs[QCOM_ENCODING] = q_attrs[QCOM_ENCODING]
                n.args[0].meta[QCOM_REQUANTIZE] = dq_attrs

    # Dequant all the fold_quant parameters back to fp32.
    # If an operation is not supported by QNN and got fallback, it will expect a fp32 param.
    def _dequant_fold_params(self, n, quant_attrs, param):
        if quant_attrs[QCOM_ENCODING] in [
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
        ]:
            dim, axis = param.dim(), quant_attrs["axis"]
            scales = self._expand(quant_attrs[QCOM_SCALES], dim, axis)
            offsets = self._expand(quant_attrs[QCOM_ZERO_POINTS], dim, axis)
            param = param.sub(offsets).mul(scales).to(torch.float32).contiguous()
            set_parameter(param, n.args[0], self.edge_program)
        else:
            scale = quant_attrs["scale"]
            offset = quant_attrs["zero_point"]
            param = param.sub(offset).mul(scale).to(torch.float32).contiguous()
            set_parameter(param, n.args[0], self.edge_program)

        n.args[0].meta["val"] = param

    def _annotate_quant_attrs(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        # Keep track of const params that has been dequant, so it does not get
        # dequant multiple times if the const param has more than 1 user
        visited_const_param = set()
        for n in graph_module.graph.nodes:
            self._annotate_requant(n)
            # With fold_quant enabled, check if the input of dq op is quantized param.
            param = None
            if n.target in dq_ops:
                param = get_parameter(n.args[0], self.edge_program)
            if n.target not in q_ops and param is None:
                continue
            quant_attrs = get_quant_attrs(self.edge_program, n)
            self._annotate_source_nodes(n, quant_attrs)

            if param is not None and n.args[0] not in visited_const_param:
                visited_const_param.add(n.args[0])
                self._dequant_fold_params(n, quant_attrs, param)

        return graph_module

    def call(self, graph_module: torch.fx.GraphModule):
        self._annotate_quant_attrs(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
