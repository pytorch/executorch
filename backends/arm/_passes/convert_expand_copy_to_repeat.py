# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch.fx
from executorch.backends.arm.tosa_mapping import extract_tensor_meta
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class ConvertExpandCopyToRepeatPass(ExportPass):
    """
    Replace expand copy with repeat since it is a repeat that can only repeat singleton dimensions.
    """

    expand_copy = exir_ops.edge.aten.expand_copy.default
    repeat = exir_ops.edge.aten.repeat.default
    patterns = [{expand_copy: 1}]

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        partitions = get_source_partitions(
            graph, [torch.expand_copy, torch.Tensor.expand, "expand"]
        )
        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                assert len(src_partition.nodes) == 1

                expand_node = src_partition.nodes[0]
                _, shape, _ = extract_tensor_meta(expand_node.all_input_nodes[0].meta)
                multiples = cast(tuple[int], expand_node.args[1])
                expanded_rank = len(multiples)

                # Expanded shape is 'shape' front-padded with ones.
                padding = expanded_rank - len(shape)
                extended_shape = [
                    shape[i] if i >= 0 else 1 for i in range(-padding, len(shape))
                ]

                # To convert expand arg to repeat arg, non-repeated dims should have
                # multiples[dim] = 1.
                multiples = [
                    multiples[i] if extended_shape[i] == 1 else 1
                    for i in range(expanded_rank)
                ]
                args = (expand_node.args[0], multiples)

                with graph_module.graph.inserting_before(expand_node):
                    repeat_node = graph.create_node("call_function", self.repeat, args)
                    repeat_node.meta = expand_node.meta
                    for user in expand_node.users.copy():
                        user.replace_input_with(expand_node, repeat_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
