# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .utils import dq_ops, get_quant_attrs, q_ops


class AnnotateDecomposed(ExportPass):
    """
    Add "quant_attrs" to graph nodes' meta from the QDQ information
    generated after quantization process.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(AnnotateDecomposed, self).__init__()
        self.edge_program = edge_program

    def _annotate_unbind(self, graph_module: torch.fx.GraphModule):
        partitions = get_source_partitions(graph_module.graph, [torch.unbind, "unbind"])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                if src_partition.input_nodes[0].target in dq_ops:
                    q_node = src_partition.input_nodes[0].args[0]
                    quant_attrs = get_quant_attrs(self.edge_program, q_node)
                    for n in src_partition.nodes:
                        n.meta[QCOM_QUANT_ATTRS] = quant_attrs.copy()

    def _annotate_stack(self, graph_module: torch.fx.GraphModule):
        partitions = get_source_partitions(graph_module.graph, [torch.stack])
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                output = src_partition.output_nodes[0]
                if (list(output.users)[0].target) in q_ops:
                    quant_attrs = get_quant_attrs(
                        self.edge_program, list(output.users)[0]
                    )
                    for n in src_partition.nodes:
                        n.meta[QCOM_QUANT_ATTRS] = quant_attrs.copy()

    def call(self, graph_module: torch.fx.GraphModule):
        self._annotate_unbind(graph_module)
        self._annotate_stack(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
