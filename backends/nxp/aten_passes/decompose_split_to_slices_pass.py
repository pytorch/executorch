# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, TypeAlias

import torch
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class DecomposeSplitToSlicesPass(PassBase):
    """
      The `split` operator returns multiple tensors by partitioning `x` along `dim`. Each partitioning can be done
      using one `slice` operator. Replacing the `split` operator with multiple `slice` operators will yield the same results.


                                  │
                    ┌─────────────▼─────────────┐
                    │             x             │
                    └─────────────┬─────────────┘
                                  │
            ┌─────────────────────▼─────────────────────┐
            │     aten.split / aten.split_with_sizes    │
            └─────────────────────┬─────────────────────┘
                                  │
             ┌────────────────────┼─────────────────────────┐
             │                    │                         │
    ┌────────▼────────┐  ┌────────▼────────┐       ┌────────▼────────┐
    │    getitem(0)   │  │    getitem(1)   │  ...  │   getitem(N-1)  │
    └────────┬────────┘  └────────┬────────┘       └────────┬────────┘
             │                    │                         │
             ▼                    ▼                         ▼
            out0                 out1                    out(N-1)


                                  |
                                  |
                             replace with
                                  |
                                  |
                                  ▼


                                  │
                    ┌─────────────▼─────────────┐
                    │             x             │
                    └─────────────┬─────────────┘
                                  │
             ┌────────────────────┼─────────────────────────┐
             │                    │                         │
    ┌────────▼────────┐  ┌────────▼────────┐       ┌────────▼────────┐
    │  aten.slice(x,  │  │  aten.slice(x,  │  ...  │  (more slices)  │
    │    dim,s0,e0    │  │   dim,s1,e1)    │  ...  │                 │
    └────────┬────────┘  └────────┬────────┘       └────────┬────────┘
             │                    │                         │
             │                    │                         │
             ▼                    ▼                         ▼
            out0                 out1                     outN-1

    """

    graph_module: GraphModule

    @staticmethod
    def _is_split_with_sizes(node: Node) -> bool:
        return (
            node.op == "call_function"
            and node.target == torch.ops.aten.split_with_sizes.default
        )

    @staticmethod
    def _is_regular_split(node: Node) -> bool:
        is_split_tensor = (
            node.op == "call_function" and node.target == torch.ops.aten.split.Tensor
        )

        is_split_default = (
            node.op == "call_function" and node.target == torch.ops.aten.split.default
        )

        return is_split_tensor or is_split_default

    def _create_slice_node(self, *slice_args) -> Node:
        slice_target = torch.ops.aten.slice.Tensor
        slice_node = self.graph_module.graph.call_function(slice_target, slice_args)

        slice_node.meta["source_fn_stack"] = [
            (slice_node.name, torch.ops.aten.slice.Tensor)
        ]

        with FakeTensorMode() as mode:
            input_ = slice_args[0].meta["val"]

            fake_input = FakeTensor.from_tensor(
                torch.empty(input_.shape, dtype=input_.dtype), mode
            )
            output = slice_target(fake_input, *slice_args[1:])
            slice_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output.shape, dtype=output.dtype), mode
            )

        return slice_node

    SlicesArgs: TypeAlias = tuple[list[int], list[int], int]

    def _get_slices_args(self, split_node: Node) -> SlicesArgs:
        split_nodes_chunks = split_node.meta["val"]
        dim = 0 if len(split_node.args) < 3 else split_node.args[2]

        # Sometimes chunks are in tuples
        if isinstance(split_nodes_chunks, tuple):
            split_nodes_chunks = list(split_nodes_chunks)

        if not isinstance(split_nodes_chunks, list):
            raise RuntimeError("Faulty split chunks")

        # Get slices start, end params
        starts = []
        ends = []

        curr_start = 0
        for s in split_nodes_chunks:
            starts.append(curr_start)
            ends.append(curr_start + s.shape[dim])
            curr_start += s.shape[dim]

        return starts, ends, dim

    def _replace_split_with_slices(self, input_node, split_node, starts, ends, dim):
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

    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        self.graph_module = graph_module
        made_changes = False

        if not any(map(self._is_regular_split, graph_module.graph.nodes)) and not any(
            map(self._is_split_with_sizes, graph_module.graph.nodes)
        ):
            return PassResult(graph_module, made_changes)

        for node in graph_module.graph.nodes:
            # Skip if not split
            is_split_with_sizes = self._is_split_with_sizes(node)
            is_regular_split = self._is_regular_split(node)

            if not is_split_with_sizes and not is_regular_split:
                continue

            # Get split args
            split_node = node
            input_node = split_node.all_input_nodes[0]
            split_nodes_chunks = split_node.meta["val"]

            # Check if split is even necessary - if not, remove it
            if len(split_nodes_chunks) == 1:
                getitem_node = list(split_node.users)[0]
                getitem_node.replace_all_uses_with(input_node)

                self.graph_module.graph.erase_node(getitem_node)
                self.graph_module.graph.erase_node(split_node)

                made_changes = True
                continue

            # Get arguments for the new slices
            starts, ends, dim = self._get_slices_args(split_node)

            # Replace split with slices and restructure the graph
            self._replace_split_with_slices(input_node, split_node, starts, ends, dim)
            made_changes = True

        self.graph_module.recompile()
        self.graph_module.graph.eliminate_dead_code()

        return PassResult(self.graph_module, made_changes)
