# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import operator

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class ReduceDynamicRange(ExportPass):
    """
    Due to limitation in Qnn, we need to change torch.finfo(torch.float32).min
    to the smallest representable value in quantization.
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

    def __init__(self):
        super(ReduceDynamicRange, self).__init__()

    def _traverse_binary_node(self, graph_module: torch.fx.GraphModule):
        src_partitions = get_source_partitions(
            graph_module.graph, self.binary_op_sources
        )
        src_partitions = list(itertools.chain(*src_partitions.values()))
        for src_partition in src_partitions:
            if len(src_partition.input_nodes) == 1:
                binary_node = src_partition.nodes[0]
                # (input node 0, constant value)
                args_list = list(binary_node.args)
                # Due to limitation in Qnn, we need to change torch.finfo(torch.float32).min
                # to the smallest representable value in quantization
                for i, arg in enumerate(args_list):
                    if arg == torch.finfo(torch.float32).min:
                        args_list[i] = -255.0
                binary_node.args = tuple(args_list)

    def call(self, graph_module: torch.fx.GraphModule):
        self._traverse_binary_node(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
