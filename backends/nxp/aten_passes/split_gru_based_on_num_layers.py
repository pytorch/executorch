# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class SplitGRUBasedOnNumLayers(PassBase):
    """Replace an `aten.gru.input` operator with `num_layers > 1` with a subgraph consisting of multiple chained
             `aten.gru.input` operators with `num_layers == 1`, according to the following schematic.


                                                                          X                H_h
                                                                          │                 │
                                                                          │             ┌───▼───┐
                                                                          │             │ Split │
                                                                          │             └─┬───┬─┘
                                                                          │       ┌───────┘   └───────┐
                                                                          │ ┌─────▼──────┐     ┌──────▼──────────────────┐
                                                                          │ │ GetItem[0] │ ... │ GetItem[<num_layers>-1] │
                                                                          │ └─────┬──────┘     └──────────────┬──────────┘
                              X  X_h                                      └────┐  │                           │
                              │   │                                      ┌─────▼──▼─────┐                     │
                   ┌──────────▼───▼──────────┐                       W11─►     GRU      ◄─B11                 │
               W11─►                         ◄─B11                   W12─► num_layers=1 ◄─B12                 │
               W12─►           GRU           ◄─B12                       └───────┬──────┘                     │
               ... │ num_layers=<num_layers> │ ...                       ┌───────┴───────┐                    │
    W<num_layers>2─►                         ◄─B<num_layers>2      ┌─────▼──────┐ ┌──────▼─────┐              │
                   └────────────┬────────────┘                     │ GetItem[0] │ │ GetItem[1] │              │
                       ┌────────┴────────┐                         └─────┬──────┘ └──────┬─────┘              │
                 ┌─────▼──────┐   ┌──────▼─────┐     replace with        │        ┌──────┘                    │
                 │ GetItem[0] │   │ GetItem[1] │      ─────────►         └────────┼─────── ... ────────────┐  │
                 └─────┬──────┘   └──────┬─────┘                                  │                  ┌─────▼──▼─────┐
                       ▼                 ▼                                        │   W<num_layers>1─►     GRU      ◄─B<num_layers>1
                       Y                Y_h                                       │   W<num_layers>2─► num_layers=1 ◄─B<num_layers>2
                                                                                  │                  └──────┬───────┘
                                                                                  │                 ┌───────┴───────┐
                                                                                  │           ┌─────▼──────┐ ┌──────▼─────┐
                                                                                  │           │ GetItem[0] │ │ GetItem[1] │
                                                                                  │           └─────┬──────┘ └──────┬─────┘
                                                                                  │ ...  ┌──────────┼───────────────┘
                                                                                 ┌▼──────▼┐         │
                                                                                 │ Concat │         │
                                                                                 └───┬────┘         │
                                                                                     ▼              ▼
                                                                                    Y_h             Y

        The `aten.gru.input` has the following schema:
             aten::gru.input(
                 Tensor input, Tensor hx, Tensor[] params,
                 bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first
             ) -> (Tensor, Tensor)
    """

    module: GraphModule

    def _get_topologically_last_node(self, nodes: list[Node]) -> Node:
        """Return the node from `nodes` which appears last in the graph."""
        for node in reversed(self.module.graph.nodes):
            if node in nodes:
                return node

        raise RuntimeError(f"None of the nodes `{nodes}` are in the graph.")

    def _create_and_insert_get_item_node(self, input_node: Node, idx: int) -> Node:
        """Create a `GetItem` node which extracts the output of `input_node` on index `idx`.
        The `GetItem` is also added to the graph right after the `input_node`.
        """
        with self.module.graph.inserting_after(input_node):
            get_item_node = self.module.graph.create_node(
                "call_function",
                operator.getitem,
                (input_node, idx),
                {},
            )

            # Assign the `source_fn_stack` and `val` meta fields as they are required for quantization.
            get_item_node.meta["source_fn_stack"] = [
                (get_item_node.name, input_node.meta["source_fn_stack"])
            ]
            get_item_node.meta["val"] = input_node.meta["val"][idx]

        return get_item_node

    def _create_gru_node(self, *gru_args) -> Node:
        """Create an `aten.gru.input` node with the provided arguments. The node will NOT be added to the graph
             automatically.

        :param gru_args: Arguments for the `aten.gru.input` operation.
        :return: The created GRU Node.
        """
        gru_target = torch.ops.aten.gru.input
        gru_node = self.module.graph.call_function(gru_target, gru_args)

        # Assign the `source_fn_stack` and `val` meta fields as they are required for quantization.
        gru_node.meta["source_fn_stack"] = [(gru_node.name, torch.nn.modules.rnn.GRU)]

        # Compute the shapes of the GRU outputs, and assign the `val` meta.
        x_val, h_val = gru_args[0].meta["val"], gru_args[1].meta["val"]
        with FakeTensorMode() as mode:
            fake_x = FakeTensor.from_tensor(
                torch.empty(x_val.shape, dtype=x_val.dtype), mode
            )
            fake_h = FakeTensor.from_tensor(
                torch.empty(h_val.shape, dtype=h_val.dtype), mode
            )
            fake_weights = [
                FakeTensor.from_tensor(
                    torch.empty(w.meta["val"].shape, dtype=x_val.dtype), mode
                )
                for w in gru_args[2]
            ]
            output_shapes = [
                t.shape for t in gru_target(fake_x, fake_h, fake_weights, *gru_args[3:])
            ]
            gru_node.meta["val"] = tuple(
                [
                    FakeTensor.from_tensor(torch.empty(shape, dtype=h_val.dtype), mode)
                    for shape in output_shapes
                ]
            )

        return gru_node

    def _create_split_node(self, *split_args) -> Node:
        """Create an `aten.split.default` node with the provided arguments. The node will NOT be added to the graph
             automatically.

        :param split_args: Arguments for the `aten.split.default` operation.
        :return: The created Split Node.
        """
        split_target = torch.ops.aten.split.default
        split_node = self.module.graph.call_function(split_target, split_args)

        # Assign the `source_fn_stack` and `val` meta fields as they are required for quantization.
        split_node.meta["source_fn_stack"] = [(split_node.name, torch.split)]

        # Compute the output shapes for the `split`, and assign the `val` meta.
        x_val = split_args[0].meta["val"]
        with FakeTensorMode() as mode:
            fake_input = FakeTensor.from_tensor(
                torch.empty(x_val.shape, dtype=x_val.dtype), mode
            )
            output_shapes = [t.shape for t in split_target(fake_input, *split_args[1:])]
            split_node.meta["val"] = tuple(
                [
                    FakeTensor.from_tensor(torch.empty(shape, dtype=x_val.dtype), mode)
                    for shape in output_shapes
                ]
            )

        return split_node

    def create_concat_node(self, *cat_args) -> Node:
        """Create an `aten.cat.default` node with the provided arguments. The node will NOT be added to the graph
             automatically.

        :param cat_args: Arguments for the `aten.cat.default` operation.
        :return: The created Cat Node.
        """
        cat_target = torch.ops.aten.cat.default
        cat_node = self.module.graph.call_function(cat_target, cat_args)

        # Assign the `source_fn_stack` and `val` meta fields as they are required for quantization.
        cat_node.meta["source_fn_stack"] = [(cat_node.name, torch.cat)]

        # Compute the output shape for the `concat`, and assign the `val` meta.
        with FakeTensorMode() as mode:
            fake_inputs = [
                FakeTensor.from_tensor(
                    torch.empty(
                        input_.meta["val"].shape, dtype=input_.meta["val"].dtype
                    ),
                    mode,
                )
                for input_ in cat_args[0]
            ]
            output = cat_target(fake_inputs, *cat_args[1:])
            cat_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output.shape, dtype=output.dtype), mode
            )

        return cat_node

    def call(self, module: GraphModule) -> PassResult:
        self.module = module
        made_changes = False

        def _is_gru(node_: Node) -> bool:
            return (
                node_.op == "call_function" and node_.target == torch.ops.aten.gru.input
            )

        if not any(map(_is_gru, module.graph.nodes)):
            return PassResult(module, False)  # No GRU nodes in the model.

        for node in module.graph.nodes:
            if not _is_gru(node):
                continue  # Not GRU.

            original_gru_node = node
            if (num_layers := original_gru_node.args[4]) == 1:
                # Basic 1-layer GRU.
                continue

            if (dropout := node.args[5]) != 0.0 or (train := node.args[6]):
                # Conversion for these cases is not supported, so the pre-processing should not be applied.
                continue

            # The `hx` (initial hidden state) has shape:
            #   [D * num_layers, hidden_size] or [D * num_layers, batch_size, hidden_size]
            #    where D = 2 if bidirectional else 1.
            # Split the `hx` into `num_layers` different tensors.
            h_x = original_gru_node.args[1]
            bidirectional = original_gru_node.args[7]
            d = 2 if bidirectional else 1
            with module.graph.inserting_before(original_gru_node):
                # Split across the dimension `0`. Slices of size `d`.
                num_slices = h_x.meta["val"].shape[0] // d
                split_node = self._create_split_node(h_x, [d] * num_slices, 0)

            # Add `GetItem` nodes to extract the outputs of the `split_node`.
            h_0_get_item_nodes = [
                self._create_and_insert_get_item_node(split_node, i)
                for i in range(num_layers)
            ]

            # ---- Create new GRU nodes ----

            all_weights = original_gru_node.args[2]
            has_biases = original_gru_node.args[3]
            # The `all_weights` list contains
            #  [w11, w12, b11, b12, w21, w22, b21, b22, ...] if `has_biases` else [w11, w12, w21, w22, ...].
            step = 4 if has_biases else 2
            if bidirectional:
                # Every other set of weights and biases (2 or 4) represents the reverse connections for the layer.
                step *= 2

            gru_nodes = []
            batch_first = original_gru_node.args[-1]

            # The `GetItem` node which extracts the main output (y) of the previous GRU. (Or the main input for the
            #  first GRU).
            prev_gru_main_output_get_item = original_gru_node.args[0]
            output_h_get_item_nodes = (
                []
            )  # `GetItem` nodes which extract the output hidden states of the GRU nodes.
            for i in range(num_layers):
                current_gru_weights = tuple(all_weights[step * i : step * (i + 1)])

                # Select the node to insert the new `GRU` after.
                prev_node = (
                    self._get_topologically_last_node(h_0_get_item_nodes)
                    if i == 0
                    else prev_gru_main_output_get_item
                )

                # Create the new `GRU`.
                with module.graph.inserting_after(prev_node):
                    gru = self._create_gru_node(
                        prev_gru_main_output_get_item,
                        h_0_get_item_nodes[i],
                        current_gru_weights,
                        has_biases,
                        1,  # New `num_layers`.
                        dropout,
                        train,
                        bidirectional,
                        batch_first,
                    )
                    gru_nodes.append(gru)

                # Create the `GetItem` nodes to extract the outputs of the `GRU`.
                prev_gru_main_output_get_item = self._create_and_insert_get_item_node(
                    gru_nodes[i], 0
                )
                output_h_get_item_nodes.append(
                    self._create_and_insert_get_item_node(gru_nodes[i], 1)
                )

            # Add a `Concat` to collect all the output hidden states.
            with module.graph.inserting_after(prev_gru_main_output_get_item):
                concat_node = self.create_concat_node(
                    output_h_get_item_nodes, 0  # Concatenate along the dimension `0`.
                )

            # Replace the uses of the original `GRU` outputs with the new corresponding outputs.
            original_y_get_item, original_yh_get_item = list(
                original_gru_node.users.keys()
            )
            original_y_get_item.replace_all_uses_with(prev_gru_main_output_get_item)
            original_yh_get_item.replace_all_uses_with(concat_node)

            # Remove the old nodes.
            module.graph.erase_node(original_y_get_item)
            module.graph.erase_node(original_yh_get_item)
            module.graph.erase_node(original_gru_node)

            made_changes = True

        return PassResult(module, made_changes)
