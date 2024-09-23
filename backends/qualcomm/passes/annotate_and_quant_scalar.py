# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import operator
from typing import Dict

import torch
from executorch.backends.qualcomm.builders.utils import get_parameter
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .utils import dq_ops, get_quant_attrs


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
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.div.Scalar,
        "add",
        "sub",
        "mul",
        "truediv",
    ]

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(AnnotateAndQuantScalar, self).__init__()
        self.edge_program = edge_program

    def _get_source_scalar_node(self, node: torch.fx.Node) -> torch.fx.Node:
        """
        This recursion function is specific for multiply followed by a cast
        """
        if node.op == "placeholder":
            if not (shape := node.meta["val"].size()):
                return node
            assert f"The output of node {node} is not a scalar, but a tensor with shape {shape}"
        return self._get_source_scalar_node(node.args[0])

    def _update_scalar_node_attrs(self, node: torch.fx.Node, quant_attrs: Dict) -> Dict:
        val = get_parameter(node, self.edge_program)
        quant_range = quant_attrs["quant_max"] - quant_attrs["quant_min"]
        # Use 0 as the zero_point for scalar
        quant_attrs["zero_point"] = 0 if val >= 0 else quant_attrs["quant_max"]
        quant_attrs["scale"] = (
            val.div(quant_range) if val >= 0 else -val.div(quant_range)
        )
        return quant_attrs

    def _annotate_scalar_node(
        self,
        be_annotated_node: torch.fx.Node,
        quant_attrs: Dict,
    ) -> None:
        """
        This recursion function is specific for multiply followed by a cast
        """
        if be_annotated_node.meta["val"].dtype not in [
            float,
            torch.float32,
            torch.int32,
            torch.int64,
        ]:
            return

        be_annotated_node.meta[QCOM_QUANT_ATTRS] = quant_attrs

    def _traverse_binary_node(self, graph_module: torch.fx.GraphModule):
        src_partitions = get_source_partitions(
            graph_module.graph, self.binary_op_sources
        )
        src_partitions = list(itertools.chain(*src_partitions.values()))
        processed = set()
        for src_partition in src_partitions:
            # need post process here to identify partitioned nodes:
            src_fn_dict = {}
            for n in src_partition.nodes:
                # e.g.
                # meta["source_fn_stack"]: [('mul', <built-in function mul>)]
                # we'll use <built-in function mul> as grouping key
                node_list = src_fn_dict.setdefault(n.meta["source_fn_stack"][-1][1], [])
                node_list.append(n)

            for nodes in src_fn_dict.values():
                output = [n for n in nodes if n in src_partition.output_nodes][0]
                # if all args have been annotated, it shouldn't be a scalar operation
                if all(arg.target in dq_ops for arg in output.args):
                    continue

                if output not in processed and QCOM_QUANT_ATTRS in output.meta:
                    dq_node = [n for n in output.args if n.target in dq_ops][0]
                    q_node = dq_node.args[0]
                    q_node_attrs = get_quant_attrs(graph_module, q_node)

                    scalar_nodes = [n for n in output.args if n != dq_node]
                    if len(scalar_nodes) == 0:
                        continue

                    scalar_node = scalar_nodes[0]
                    source_scalar_node = self._get_source_scalar_node(scalar_node)
                    # we'll abandon cast op here, since the constant scalar will
                    # be pre-loaded into QNN context binary
                    output.replace_input_with(scalar_node, source_scalar_node)

                    scalar_quant_attrs = self._update_scalar_node_attrs(
                        source_scalar_node, q_node_attrs
                    )
                    self._annotate_scalar_node(source_scalar_node, scalar_quant_attrs)
                    processed.add(output)

    def call(self, graph_module: torch.fx.GraphModule):
        self._traverse_binary_node(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
