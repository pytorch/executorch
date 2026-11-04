# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_ops
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from .utils import get_quant_attrs


class AnnotateUnbind(ExportPass):
    """
    Add "quant_attrs" to graph nodes' meta from the QDQ information
    generated after quantization process.
    """

    _SOURCE_OPS = [torch.unbind, torch.ops.aten.unbind.int, "unbind"]

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(AnnotateUnbind, self).__init__()
        self.edge_program = edge_program

    def _annotate_unbind(self, graph_module: torch.fx.GraphModule):
        partitions = get_source_partitions(graph_module.graph, self._SOURCE_OPS)
        for src_partitions in partitions.values():
            for src_partition in src_partitions:
                dq_node = src_partition.input_nodes[0]
                if dq_node.target in dq_ops:
                    # a folded quantized constant feeds its dequantize directly
                    q_node = dq_node.args[0]
                    if q_node.target not in q_ops:
                        q_node = dq_node
                    quant_attrs = get_quant_attrs(self.edge_program, q_node)
                    for n in src_partition.nodes:
                        n.meta[QCOM_QUANT_ATTRS] = quant_attrs.copy()

    def call(self, graph_module: torch.fx.GraphModule):
        self._annotate_unbind(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
