# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm._passes.fold_qdq import FoldQDQ
from executorch.backends.qualcomm._passes.utils import get_quant_attrs
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.builders.utils import (
    is_graph_input,
    is_graph_output,
    is_parameter,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_BYPASS_NODE,
    QCOM_FALLBACK_NODE,
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
)


class LpaiFoldQDQ(FoldQDQ):
    """
    LPAI-specific extension of FoldQDQ.

    In LPAI backend v6, there is an accuracy drop for the quantize and
    dequantize operations. To address this, keep the quantize/dequantize
    operations at the model's input and output.

    For example:
        input -> q_1 (Fallback) -> dq_1 (Bypass) -> graph -> q_2 (Bypass) -> dq_2 (Fallback) -> output

    Here, q_1 and dq_2 will fallback to CPU, while q_2 and dq_1 will be
    bypassed in qnn_partition and folded in qnn_preprocess.
    """

    def _preserve_qdq(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            # skip parameters & buffers (base class logic)
            if n.target in dq_ops and is_parameter(n.args[0], self.edge_program):
                self._annotate_bypass(n)
                continue

            if (
                is_graph_input(n, self.edge_program)
                # For tagged quantized I/O, we should not fallback quantize operation.
                and QCOM_QUANTIZED_IO not in n.meta
            ):
                user_list = list(n.users.keys())
                if len(user_list) > 0:
                    q_node = user_list[0]
                    q_node.meta[QCOM_FALLBACK_NODE] = True
                    # Annotate the q_node since it will serve as the input for the first node during operator validation
                    q_node.meta[QCOM_QUANT_ATTRS] = get_quant_attrs(
                        self.edge_program, q_node
                    )
                    q_node.meta[QCOM_QUANTIZED_IO] = q_node.args[-1]
                    dq_node = list(q_node.users.keys())[0]
                    # Bypass dequantize op for graph validation by torch
                    dq_node.meta[QCOM_BYPASS_NODE] = True
                    # Make sure that the quantize operator isn't inserted for input in insert_io_qdq.py
                    n.meta[QCOM_QUANTIZED_IO] = q_node.args[-1]
            elif (
                is_graph_output(n)
                and n.target in dq_ops
                # For tagged quantized I/O, we should not fallback dequantize operation.
                and QCOM_QUANTIZED_IO not in n.args[0].args[0].meta
            ):
                n.meta[QCOM_FALLBACK_NODE] = True
                q_node = n.args[0]
                # Bypass quantize op for graph validation by torch
                q_node.meta[QCOM_BYPASS_NODE] = True
                op_node = q_node.args[0]
                # Make sure that the dequantize operator isn't inserted for output in insert_io_qdq.py
                op_node.meta[QCOM_QUANTIZED_IO] = q_node.args[-1]
