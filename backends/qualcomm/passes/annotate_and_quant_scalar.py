# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import operator
from typing import Dict

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .annotate_quant_attrs import get_quant_attrs


class AnnotateAndQuantScalar(ExportPass):
    """
    For binary operators who take constant scalar as one of its inputs,
    will annotate encoding to the constant if necessary.
    """

    binary_op_sources = [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
        "add",
        "sub",
        "mul",
        "truediv",
    ]
    quant_attrs_key = "quant_attrs"

    def __init__(self):
        super(AnnotateAndQuantScalar, self).__init__()

    def _get_source_scalar_node(self, node: torch.fx.Node) -> torch.fx.Node:
        """
        This recursion function is specific for multiply followed by a cast
        """
        if node.op == "get_attr":
            if not (shape := node.meta["val"].size()):
                return node
            assert f"The output of node {node} is not a scalar, but a tensor with shape {shape}"
        return self._get_source_scalar_node(node.args[0])

    def _update_scalar_node_attrs(
        self, gm: torch.fx.GraphModule, node: torch.fx.Node, quant_attrs: Dict
    ) -> Dict:
        scalar_name = node.name
        val = getattr(gm, scalar_name)
        # Use 0 as the zero_point for scalar
        quant_attrs["zero_point"] = 0
        quant_attrs["scale"] = val.div(
            quant_attrs["quant_max"] - quant_attrs["quant_min"]
        )
        return quant_attrs

    def _annotate_scalar_node(
        self,
        gm: torch.fx.GraphModule,
        be_annotated_node: torch.fx.Node,
        quant_attrs: Dict,
    ) -> None:
        """
        This recursion function is specific for multiply followed by a cast
        """
        if be_annotated_node.meta["val"].dtype not in [float, torch.float32]:
            return

        be_annotated_node.meta[self.quant_attrs_key] = quant_attrs
        if be_annotated_node.op != "get_attr":
            self._annotate_scalar_node(gm, be_annotated_node.args[0], quant_attrs)

    def _traverse_binary_node(self, graph_module: torch.fx.GraphModule):
        src_partitions = get_source_partitions(
            graph_module.graph, self.binary_op_sources
        )
        src_partitions = list(itertools.chain(*src_partitions.values()))
        for src_partition in src_partitions:
            output = src_partition.output_nodes[0]
            if (
                output.meta.get(self.quant_attrs_key)
                and len(src_partition.input_nodes) == 1
            ):
                dq_node = src_partition.input_nodes[0]
                q_node = dq_node.args[0]
                q_node_attrs = get_quant_attrs(graph_module, q_node)

                scalar_node = [n for n in output.args if n != dq_node][0]
                source_scalar_node = self._get_source_scalar_node(scalar_node)

                scalar_quant_attrs = self._update_scalar_node_attrs(
                    graph_module, source_scalar_node, q_node_attrs
                )
                self._annotate_scalar_node(
                    graph_module, scalar_node, scalar_quant_attrs
                )

    def call(self, graph_module: torch.fx.GraphModule):
        self._traverse_binary_node(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
