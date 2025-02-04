# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_ENCODING, QCOM_QUANT_ATTRS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ReplaceIndexPutInput(ExportPass):
    """
    Index put input workaround for quantized module
    """

    dq_q_map = {
        # per tensor
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default: exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor: exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        # per channel
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default: exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    }

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(ReplaceIndexPutInput, self).__init__()
        self.edge_program = edge_program

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target == exir_ops.edge.aten.index_put.default:
                if (
                    copy_node := list(node.users)[0]
                ) and copy_node.target == exir_ops.edge.aten.copy.default:
                    m_buffer_node = copy_node.args[0]
                    bad_frozen_node = node.args[0]
                    if QCOM_QUANT_ATTRS in bad_frozen_node.meta:
                        m_buffer_node.meta[QCOM_QUANT_ATTRS] = bad_frozen_node.meta[
                            QCOM_QUANT_ATTRS
                        ]
                        m_buffer_node.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING] = (
                            self.dq_q_map[
                                m_buffer_node.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING]
                            ]
                        )
                    with graph.inserting_after(bad_frozen_node):
                        node.replace_input_with(bad_frozen_node, m_buffer_node)
                else:
                    continue

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
