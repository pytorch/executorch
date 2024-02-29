# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import operator
from collections import Counter
from typing import List

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class ConvertBmmToMatmul(ExportPass):
    """
    Replace bmm to matmul, because bmm is eqaul to matmul in QNN.
    Handle missing quantization tag for bmm op.
    """

    view_copy = exir_ops.edge.aten.view_copy.default
    expand_copy = exir_ops.edge.aten.expand_copy.default
    clone = exir_ops.edge.aten.clone.default
    bmm = exir_ops.edge.aten.bmm.default
    matmul = exir_ops.edge.aten.matmul.default
    patterns = [
        {expand_copy: 2, view_copy: 3, bmm: 1},
        {expand_copy: 2, view_copy: 3, bmm: 1, clone: 1},
        {bmm: 1},
    ]

    def __init__(self):
        super(ConvertBmmToMatmul, self).__init__()

    def _get_ordered_inputs(
        self, inputs: List[torch.fx.Node], output: torch.fx.Node
    ) -> List[torch.fx.Node]:
        bmm_inputs = []
        for arg in output.args:
            while arg not in inputs:
                arg = arg.args[0]
            bmm_inputs.append(arg)
        return bmm_inputs

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(
            graph, [operator.matmul, torch.matmul, torch.bmm]
        )
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                op_cnt = Counter([n.target for n in src_partition.nodes])
                if op_cnt not in self.patterns:
                    continue

                inputs = src_partition.input_nodes
                bmm_node = [n for n in src_partition.nodes if n.target == self.bmm][0]
                output = src_partition.output_nodes[0]
                # the order of src_partition.inputs is not guaranteed.
                lhs, rhs = self._get_ordered_inputs(inputs, bmm_node)
                with graph_module.graph.inserting_before(output):
                    # replace bmm to matmul, because bmm is eqaul to matmul in qnn.
                    matmul_node = graph.create_node(
                        "call_function", self.matmul, (lhs, rhs)
                    )
                    matmul_node.meta = output.meta
                    for user in output.users.copy():
                        user.replace_input_with(output, matmul_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
