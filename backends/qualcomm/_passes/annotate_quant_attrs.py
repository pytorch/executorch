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
    QCOM_AXIS,
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_QUANT_ATTRS,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_REQUANTIZE,
    QCOM_SCALE,
    QCOM_SCALES,
    QCOM_ZERO_POINT,
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

    def __init__(
        self, edge_program: torch.export.ExportedProgram, skip_advanced_requat: bool
    ):
        super(AnnotateQuantAttrs, self).__init__()
        self.edge_program = edge_program
        self.skip_advanced_requant = skip_advanced_requat

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

    # Find the the last dq nodes between regular op nodes
    # Return dq2 in example below when q1 is given as node parameter:
    # ... -> n1 -> q1 -> dq1 -> q2 -> dq2 -> n2 -> ...
    def _find_last_dq_nodes(self, node: torch.fx.node.Node) -> torch.fx.node.Node:
        if node is None:
            return []

        # If the node is last dq between regular op node, return it in a list
        if node.target in dq_ops:
            if all(user.target not in q_ops for user in node.users):
                return [node]

        last_dq_nodes = []
        for user in list(node.users):
            last_dq_nodes.extend(self._find_last_dq_nodes(user))

        return last_dq_nodes

    def _annotate_requant(self, n):
        # Record requant attributes:
        # node1 -> q_ui8 (n) -> dq_ui8 -> q_int32 -> dq_int32 -> node2 -> ....
        # We store {node2: quant_attr in dq_int32} in node1.meta
        if n.target in q_ops and n.args[0].target not in dq_ops:
            dq_nodes = self._find_last_dq_nodes(n)
            q_attrs = get_quant_attrs(self.edge_program, n)
            for dq_node in dq_nodes:
                dq_attrs = get_quant_attrs(self.edge_program, dq_node)
                # TODO: Store multiple pairs of requantize attributes when we have an op builder
                # that has multiple outputs that requires quant attributes.
                if self.skip_advanced_requant:
                    if q_attrs[QCOM_DTYPE] != dq_attrs[QCOM_DTYPE]:
                        dq_attrs[QCOM_ENCODING] = q_attrs[QCOM_ENCODING]
                        user_node = list(dq_node.users)[0]
                        n.args[0].meta.setdefault(QCOM_REQUANTIZE, {})
                        n.args[0].meta[QCOM_REQUANTIZE][user_node.name] = dq_attrs
                else:
                    # When dtype is the same but other specs such as scale and offset are different,
                    # insert requant to improve accuracy.
                    # Users can turn this feature off if any inference speed drop is observed.
                    if any(
                        q_attrs[attr] != dq_attrs[attr]
                        for attr in [
                            QCOM_SCALE,
                            QCOM_ZERO_POINT,
                            QCOM_QUANT_MIN,
                            QCOM_QUANT_MAX,
                            QCOM_DTYPE,
                        ]
                    ):
                        dq_attrs[QCOM_ENCODING] = q_attrs[QCOM_ENCODING]
                        user_node = list(dq_node.users)[0]
                        n.args[0].meta.setdefault(QCOM_REQUANTIZE, {})
                        n.args[0].meta[QCOM_REQUANTIZE][user_node.name] = dq_attrs

    # Dequant all the fold_quant parameters back to fp32.
    # If an operation is not supported by QNN and got fallback, it will expect a fp32 param.
    def _dequant_fold_params(self, n, quant_attrs, param):
        if quant_attrs[QCOM_ENCODING] in [
            exir_ops.edge.quantized_decomposed.dequantize_per_channel.default
        ]:
            dim, axis = param.dim(), quant_attrs[QCOM_AXIS]
            scales = self._expand(quant_attrs[QCOM_SCALES], dim, axis)
            offsets = self._expand(quant_attrs[QCOM_ZERO_POINTS], dim, axis)
            param = param.sub(offsets).mul(scales).to(torch.float32).contiguous()
            set_parameter(param, n.args[0], self.edge_program)
        else:
            scale = quant_attrs[QCOM_SCALE]
            offset = quant_attrs[QCOM_ZERO_POINT]
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
