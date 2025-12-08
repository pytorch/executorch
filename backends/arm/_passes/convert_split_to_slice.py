# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
                        from_node=node,
                    )
                    slice_node.meta = _copy_user_node_qparams(
                        split_node, output_node, index
                    )
                    output_node.replace_all_uses_with(slice_node)
        graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)


def _copy_user_node_qparams(
    split_node: torch.fx.Node, output_node: torch.fx.Node, index: int
) -> dict:
    """
    Construct metadata for the slice node that will replace the split output.

    Note that output quantization parameters are copied from the user nodes
    of the split node. The split node itself does not have output quantization
    parameters.

    Args:
        split_node: The split node being replaced.
        output_node: The getitem node that is user of the split node.
        index: The index of the output being processed.
    Returns:
        Updated metadata dictionary for the slice node.
    """

    def _select_index(value):
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    meta = split_node.meta.copy()
    if "val" in meta:
        meta["val"] = _select_index(meta["val"])
    if "tensor_meta" in meta:
        meta["tensor_meta"] = _select_index(meta["tensor_meta"])
    if "input_qparams" in meta:
        meta["input_qparams"] = dict(meta["input_qparams"])
    if "output_qparams" in meta:
        meta["output_qparams"] = dict(output_node.meta["output_qparams"])
    return meta
