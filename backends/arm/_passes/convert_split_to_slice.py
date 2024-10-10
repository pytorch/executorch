# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch.fx
from executorch.backends.arm.tosa_mapping import extract_tensor_meta
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertSplitToSlicePass(ExportPass):
    """
    Replace a split operation with many slice operations.
    """

    split_ops = (
        exir_ops.edge.aten.split_with_sizes_copy.default,
        exir_ops.edge.aten.split_copy.Tensor,
    )
    slice = exir_ops.edge.aten.slice_copy.Tensor

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        for node in graph.nodes:
            if node.target not in self.split_ops:
                continue

            # Get useful variables
            split_node = node
            input_node = split_node.all_input_nodes[0]
            output_nodes = split_node.users.copy()
            _, shape, _ = extract_tensor_meta(input_node.meta)
            rank = len(shape)
            split_lengths = split_node.args[1]
            dim = split_node.args[2] if len(split_node.args) > 2 else 0
            dim = (dim + rank) % rank

            assert (
                sum(split_lengths) == shape[dim]
            ), "Given split lengths don't sum up to the size of the dimension."

            # Convert split argument 'split_lengths' to slice arguments start and end.
            starts = [0] * len(split_lengths)
            ends = [0] * len(split_lengths)
            start = 0
            end = 0
            for i, split_length in enumerate(split_lengths):
                end = start + split_length
                starts[i] = start
                ends[i] = end
                start = end

            # Output nodes are of type getitem
            # Create one slice node for each output node with matching argumetns.
            with graph_module.graph.inserting_before(split_node):
                for output_node in output_nodes:
                    index = output_node.args[1]
                    slice_node = graph.create_node(
                        "call_function",
                        self.slice,
                        (input_node, dim, starts[index], ends[index]),
                    )
                    slice_node.meta = split_node.meta.copy()
                    slice_node.meta["val"] = slice_node.meta["val"][index]
                    output_node.replace_input_with(split_node, slice_node)
        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
