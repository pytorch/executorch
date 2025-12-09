# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch._subclasses import FakeTensor, FakeTensorMode


class DecomposeSplitToSlicesPass(PassBase):
    graph_module: GraphModule

    def _get_topologically_last_node(self, nodes: list[Node]) -> Node:
        """Return the node from `nodes` which appears last in the graph."""
        for node in reversed(self.graph_module.graph.nodes):
            if node in nodes:
                return node

        raise RuntimeError(f"None of the nodes `{nodes}` are in the graph.")

    def _create_slice_node(self, *slice_args) -> Node:
        slice_target = torch.ops.aten.slice_copy.Tensor 
        slice_node = self.graph_module.graph.call_function(slice_target, slice_args)

        slice_node.meta["source_fn_stack"] = [(slice_node.name, torch.slice_copy)]

        x_val = slice_args[0].meta["val"]
        with FakeTensorMode() as mode:
            fake_input = FakeTensor.from_tensor(
                torch.empty(x_val.shape, dtype=x_val.dtype), mode
            )
            output_shape = slice_target(fake_input, *slice_args[1:]).shape
            slice_node.meta["val"] = FakeTensor.from_tensor(torch.empty(output_shape, dtype=x_val.dtype), mode)

        return slice_node
    
    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        self.graph_module = graph_module
        made_changes = False

        def _is_split_with_sizes(node: Node) -> bool:
            return (
                node.op == "call_function"
                and node.target == torch.ops.aten.split_with_sizes.default
            )
        
        def _is_regular_split(node: Node) -> bool:
            is_split_tensor = (
                node.op == "call_function"
                and node.target == torch.ops.aten.split.Tensor
            )

            is_split_default = (
                node.op == "call_function"
                and node.target == torch.ops.aten.split.default
            )

            return is_split_tensor or is_split_default

        if not any(map(_is_regular_split, graph_module.graph.nodes)) \
            and not any(map(_is_split_with_sizes, graph_module.graph.nodes)):
            return PassResult(
                graph_module, made_changes
            )  # No split type nodes in the model.

        for node in graph_module.graph.nodes:
            is_split_with_sizes = _is_split_with_sizes(node)
            is_regular_split = _is_regular_split(node)

            if not is_split_with_sizes and not is_regular_split:
                continue

            # Get split params
            split_node = node
            input_node = split_node.all_input_nodes[0]
            input_tensor_shape = input_node.meta["val"].shape
            split_nodes_chunks = split_node.meta["val"]

            # Sometimes chunks are in tuples
            if isinstance(split_nodes_chunks, tuple):
                split_nodes_chunks = list(split_nodes_chunks)

            if not isinstance(split_nodes_chunks, list):
                raise RuntimeError("Faulty split chunks")

            # When there is only one chunk, split is redundant
            if len(split_nodes_chunks) == 1:
                getitem_node = split_node.next
                getitem_node.replace_all_uses_with(input_node)

                self.graph_module.graph.erase_node(getitem_node)
                self.graph_module.graph.erase_node(split_node)

                continue

            # Get split dim
            dim = -1
            for possible_dim in range(len(split_nodes_chunks[0].shape)):
                if split_nodes_chunks[0].shape[possible_dim] != input_tensor_shape[possible_dim]:
                    dim = possible_dim
                    break

            if dim == -1:
                raise RuntimeError("Could not determine dim param")

            # Get slices start, end params
            starts = []
            ends = []

            curr_start = 0
            for s in split_nodes_chunks:
                starts.append(curr_start)
                ends.append(curr_start + s.shape[dim])
                curr_start += s.shape[dim]

            # Replace getitem nodes after split with slices
            getitem_nodes = list(split_node.users.keys())
            slice_nodes = []
            for i in range(len(starts)):
                slice_arguments = (input_node, dim, starts[i], ends[i])

                with self.graph_module.graph.inserting_after(split_node):
                    slice_node = self._create_slice_node(*slice_arguments)
                    slice_nodes.append(slice_node)

                    getitem_node = getitem_nodes[i]
                    getitem_node.replace_all_uses_with(slice_node)

                    self.graph_module.graph.erase_node(getitem_node)

            # Wire split node correctly to the input node
            split_node.replace_all_uses_with(input_node)
            self.graph_module.graph.erase_node(split_node)

            made_changes = True

        self.graph_module.recompile()
        self.graph_module.graph.eliminate_dead_code()

        return PassResult(self.graph_module, made_changes)
