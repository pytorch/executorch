# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from executorch.backends.nxp.backend.edge_helper import (
    try_get_tensor_constant_from_node,
)
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class FuseLinearAndAddPass(PassBase):
    """Replace a sequence of `linear` and `add` nodes in the following pattern by a single `linear` node when possible.
           │
    ┌──────▼──────┐
    │ aten.linear │
    └──────┬──────┘                               │
           │            replace with       ┌──────▼──────┐
     ┌─────▼────┐       ───────────►       │ aten.linear │
     │ aten.add │                          └──────┬──────┘
     └─────┬────┘
           ▼
    """

    def _fuse_with_existing_bias(
        self,
        linear_node: Node,
        other_add_input: Node,
        graph_module: GraphModule,
        alpha: float,
    ) -> bool:
        """Fuse the `linear` and `add` nodes provided the `linear` already has a bias.
         The fusion can only be done if both the "biases" have static data, which can be added together to get a
         single bias.

        :return: True, if the nodes were successfully merged. False, otherwise.
        """

        linear_bias = linear_node.args[2]
        if other_add_input.meta["val"].shape != linear_bias.meta["val"].shape:
            # The biases cannot be added together due to their different shapes.
            # Shape broadcasting is not applicable, as the only allowed `linear` bias shape is 1D ([output_features]).
            return False

        bias_data = [
            try_get_tensor_constant_from_node(graph_module, linear_bias),
            try_get_tensor_constant_from_node(graph_module, other_add_input),
        ]
        if any(data is None for data in bias_data):
            return (
                False  # Fusion is not possible because at least 1 bias is not static.
            )

        # Add the bias data together, to obtain the combined bias. Take the `alpha` attribute into account.
        combined_bias = bias_data[0] + bias_data[1] * alpha

        # Create a new node containing the combined bias data.
        combined_bias_name = get_new_attr_name_with_prefix(
            linear_bias.name + "combined"
        )(graph_module)
        _assign_attr(
            torch.nn.Parameter(combined_bias),
            graph_module,
            combined_bias_name,
            _AttrKind.PARAMETER,
        )
        with graph_module.graph.inserting_before(linear_node):
            new_bias_node = graph_module.graph.get_attr(combined_bias_name)

        # Use the combined bias as the new bias for the `Linear`.
        linear_node.args = (
            linear_node.args[:2] + (new_bias_node,) + linear_node.args[3:]
        )
        return True

    def _fuse_without_existing_bias(
        self,
        linear_node: Node,
        other_add_input: Node,
        graph_module: GraphModule,
        alpha: float,
    ) -> bool:
        """Fuse the `linear` and `add` provided the `linear` does not already have a bias.

        :return: True, if the nodes were successfully merged. False, otherwise.
        """

        # The weights have shape (out_features, in_features).
        output_features = linear_node.args[1].meta["val"].shape[0]
        new_bias_shape = other_add_input.meta["val"].shape
        if list(new_bias_shape) != [output_features]:
            return False  # The `Add` is adding a tensor with shape that is not supported for the `Linear` bias.

        bias_data = try_get_tensor_constant_from_node(graph_module, other_add_input)

        if bias_data is None:
            return False  # Neutron doesn't support a dynamic bias, so fusion would be counterproductive.

        # It is possible that the `linear` comes before the `other_add_input` in the graph, so it cannot use it as an
        #  input directly. If the nodes are ordered as [linear, ..., other_add_input, ... add] (which is valid), using
        #  `other_add_input` directly as an input to `Linear` would not follow topological order.
        # Rearranging the nodes is not trivial, as the graph could be complex (ultimately, the
        #  `other_add_input` could even originate from the `Linear` node...).
        # Since the `other_add_input` has static data, we can create a new node with the data just before the `Linear`
        #  to ensure topological order.
        # Regardless of the node ordering, the `add.Tensor` attribute `alpha` multiplies the second `add` input. If
        #  `alpha != 1`, we would have to insert a `mul` operator if we wanted to keep the original parameter node.
        #  Therefore, it is better to create a new static parameter node for the multiplied data in this case as well.
        nodes = list(graph_module.graph.nodes)
        if nodes.index(linear_node) < nodes.index(other_add_input) or alpha != 1.0:
            # Problematic order, or required multiplication.

            # Handle the `aten.add.Tensor` attribute `alpha`.
            bias_data *= alpha

            # Create a unique name.
            new_bias_name = get_new_attr_name_with_prefix(linear_node.name + "_bias")(
                graph_module
            )
            _assign_attr(bias_data, graph_module, new_bias_name, _AttrKind.PARAMETER)
            with graph_module.graph.inserting_before(linear_node):
                new_bias_node = graph_module.graph.get_attr(new_bias_name)

            # Use the added tensor as the new `Linear` bias.
            linear_node.args = (
                linear_node.args[:2] + (new_bias_node,) + linear_node.args[2:]
            )
            return True

        else:
            # Use the `other_add_input` directly as the new bias.
            linear_node.args = (
                linear_node.args[:2] + (other_add_input,) + linear_node.args[2:]
            )
            return True

    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        def _is_applicable_linear_node(node_: Node):
            is_linear = (
                node_.op == "call_function"
                and node_.target == torch.ops.aten.linear.default
            )
            has_single_user = len(node.users) == 1

            return is_linear and has_single_user

        def _is_add(node_: Node):
            return (
                node_.op == "call_function"
                and node_.target == torch.ops.aten.add.Tensor
            )

        made_changes = False
        for node in graph_module.graph.nodes:
            if not _is_applicable_linear_node(
                linear_node := node
            ):  # Also ensures a single user.
                continue

            if not _is_add(add_node := list(linear_node.users.keys())[0]):
                continue  # Not the `Linear` -> `Add` case.

            if len(add_node.args) != 2:
                continue  # Unexpected case.

            # The `aten.add.Tensor` carries out the expression `out = input[0] + alpha × input[1]`.
            # https://docs.pytorch.org/docs/stable/generated/torch.add.html
            alpha = add_node.kwargs.get("alpha", 1.0)
            if add_node.args[0] == linear_node:
                other_add_input = add_node.args[1]

            else:
                # The fusion is not implemented. The `other_add_input` would have to be divided by `alpha` before the
                #  fusion, and a `mul` operator would have to be added after the `linear` to multiply its output by
                #  `alpha`.
                continue

            if len(linear_node.args) > 2:
                if not self._fuse_with_existing_bias(
                    linear_node, other_add_input, graph_module, alpha
                ):
                    continue  # The nodes could not be fused.

            else:
                # The `Linear` doesn't have a bias yet.
                if not self._fuse_without_existing_bias(
                    linear_node, other_add_input, graph_module, alpha
                ):
                    continue  # The nodes could not be fused.

            # Use the output of the `Linear` instead of the `Add`, and remove the now unused `Add` node.
            add_node.replace_all_uses_with(linear_node)
            graph_module.graph.erase_node(add_node)

            made_changes = True

        return PassResult(graph_module, made_changes)
