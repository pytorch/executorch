# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm._passes.utils import (
    insert_dequant_node,
    insert_quant_node,
)

from executorch.backends.qualcomm.builders.node_visitor import q_dq_map, q_ops

from executorch.backends.qualcomm.builders.utils import (
    is_mutable_buffer_input,
    is_parameter,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_ENCODING,
    QCOM_QUANT_ATTRS,
    QCOM_QUANTIZED_IO,
)
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass


class InsertIOQDQ(ExportPass):
    """
    For delegated QNN subgraph, no more QDQ operators will appear after
    'fold_qdq pass'.
    This pass will insert quantize nodes right after inputs, dequantize nodes
    right before outputs according to stored quantization encodings.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        super(InsertIOQDQ, self).__init__()
        self.edge_program = edge_program

    def _insert(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # Snapshot nodes: inserting Q/DQ nodes mutates the graph's linked list,
        # so iterating the live list can revisit newly inserted nodes.
        for n in list(graph_module.graph.nodes):
            # do nothing when a node is expected to output a quant tensor
            if n.meta.get(QCOM_QUANTIZED_IO):
                continue

            # insert q after input or fold mix_quantization dq if applicable
            if (
                n.op == "placeholder"
                and n.meta.get(QCOM_QUANT_ATTRS)
                and (
                    not is_parameter(n, self.edge_program)
                    or is_mutable_buffer_input(n, self.edge_program)
                )
            ):
                # Insert a single Q node and connect single Q node to all users
                users = list(n.users.keys())
                if users:
                    input_q_node = insert_quant_node(
                        graph_module=graph_module,
                        input_node=n,
                        output_node=users[0],
                        target=n.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING],
                    )
                    for user in users[1:]:
                        if user.target not in q_ops:
                            user.replace_input_with(n, input_q_node)

            # insert dq before output or fold mix_quantization q if applicable
            users = [user for user in n.users if user.op == "output"]
            if n.meta.get(QCOM_QUANT_ATTRS) and len(users) != 0:
                # We should always only have 1 output node. Expect len(users) == 1
                for user in users:
                    _ = insert_dequant_node(
                        graph_module=graph_module,
                        input_node=n,
                        output_node=user,
                        target=q_dq_map[n.meta[QCOM_QUANT_ATTRS][QCOM_ENCODING]],
                    )

    def call(self, graph_module: torch.fx.GraphModule):
        self._insert(graph_module)
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
