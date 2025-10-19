# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Set, Type

import torch.fx
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertSplitToSlicePass(ArmPass):
    """
    Replace a split operation with many slice operations.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

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
            shape = get_first_fake_tensor(input_node).shape
            rank = len(shape)
            split_lengths = split_node.args[1]
            dim = split_node.args[2] if len(split_node.args) > 2 else 0
            dim = (dim + rank) % rank

            # Validate that split lengths cover the entire dimension

            dim_size = shape[dim]
            if isinstance(split_lengths, int):
                if split_lengths <= 0:
                    raise ValueError(
                        f"Split size must be positive, got {split_lengths}"
                    )
                full_chunks, remainder = divmod(dim_size, split_lengths)
                split_lengths = [split_lengths] * full_chunks
                if remainder:
                    split_lengths.append(remainder)
            else:
                length_sum = sum(split_lengths)
                if length_sum != dim_size:
                    raise ValueError(
                        f"Split sizes {split_lengths} sum to {length_sum}, "
                        f"but dimension {dim} has size {dim_size}"
                    )

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
            # Replace them with one slice node for each output node.
            with graph_module.graph.inserting_before(split_node):
                for output_node in output_nodes:
                    index = output_node.args[1]
                    slice_node = create_node(
                        graph,
                        self.slice,
                        (input_node, dim, starts[index], ends[index]),
                    )
                    slice_node.meta = split_node.meta.copy()
                    slice_node.meta["val"] = slice_node.meta["val"][index]
                    output_node.replace_all_uses_with(slice_node)
        graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
