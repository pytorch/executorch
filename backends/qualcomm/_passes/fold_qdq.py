# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_ops
from executorch.backends.qualcomm.builders.utils import (
    is_graph_input,
    is_graph_output,
    is_parameter,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_BYPASS_NODE,
    QCOM_FALLBACK_NODE,
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import get_quant_attrs


class FoldQDQ(ExportPass):
    """
    Erase QDQ pattern.
    """

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        force_fold=False,
        backend_type: QnnExecuTorchBackendType = QnnExecuTorchBackendType.kHtpBackend,
    ):
        super(FoldQDQ, self).__init__()
        self.edge_program = edge_program
        self.force_fold = force_fold
        self.backend_type = backend_type

    def _annotate_bypass(self, node):
        node.meta[QCOM_BYPASS_NODE] = True
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.op == "call_function":
                self._annotate_bypass(arg)

    def _fold_dq(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # remove dq
        for n in graph_module.graph.nodes:
            user_list = list(n.users.keys())
            if n.target not in dq_ops:
                continue

            if not self.force_fold and (
                QCOM_BYPASS_NODE in n.meta or QCOM_FALLBACK_NODE in n.meta
            ):
                continue

            for user_n in user_list:
                user_n.replace_input_with(n, n.args[0])
            graph_module.graph.erase_node(n)

    def _fold_q(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # remove q
        for n in graph_module.graph.nodes:
            if n.target not in q_ops:
                continue

            if not self.force_fold and (
                QCOM_BYPASS_NODE in n.meta or QCOM_FALLBACK_NODE in n.meta
            ):
                continue

            to_be_removed = [n]
            source_n = n.args[0]

            # TODO: remove this hack as source_fn_stack is internal implementation detail of torch.export.
            # To make constant value/tensor be tagged as delegatable during partition
            if source_n.op == "get_attr":
                source_n.meta["source_fn_stack"] = list(n.users.keys())[0].meta.get(
                    "source_fn_stack"
                )

            # collecting quant nodes to be removed
            for i in range(1, len(n.args)):
                if isinstance(n.args[i], torch.fx.node.Node):
                    to_be_removed.append(n.args[i])
                    # could be a commonly shared attribute between q & dq
                    if n.args[i].target == exir_ops.edge.aten._to_copy.default:
                        to_be_removed.append(n.args[i].args[0])
            # connect source node to quant users and remove quant node
            for user_n in list(n.users.keys()):
                user_n.replace_input_with(n, n.args[0])
            for n in to_be_removed:
                graph_module.graph.erase_node(n)

    def _preserve_qdq(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for n in graph_module.graph.nodes:
            # skip parameters & buffers
            if n.target in dq_ops and is_parameter(n.args[0], self.edge_program):
                self._annotate_bypass(n)
                continue

            # TODO: In LPAI backend v6, there is an accuracy drop for the quantize and dequantize operations.
            # To address this, keep the quantize/dequantize operations at the model's input and output.
            # For example, input -> q_1 (Fallback) -> dq_1 (Bypass) -> graph -> q_2 (Bypass) -> dq_2 (Fallback) -> output
            # Here, q_1 and dq_2 will fallback to CPU, while q_2 and dq_1 will be bypassed in qnn_partition and folded in qnn_preprocess.
            if self.backend_type == QnnExecuTorchBackendType.kLpaiBackend:
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

    def call(self, graph_module: torch.fx.GraphModule):
        if not self.force_fold:
            self._preserve_qdq(graph_module)
        self._fold_dq(graph_module)
        self._fold_q(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
