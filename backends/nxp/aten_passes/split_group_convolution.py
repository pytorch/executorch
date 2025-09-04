# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch

from executorch.backends.nxp.backend.ir.converter.node_converters.shared.conv_utils import (
    group_conv_convertible_into_multiple_convolutions,
)
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.nn.parameter import Parameter


class SplitGroupConvolution(PassBase):
    """The eIQ Neutron NPU supports only regular and depthwise convolutions. Group convolutions must be decomposed into
             multiple parallel single group convolutions.
           Replace the nodes in the following pattern. The square brackets indicate the tensor shapes.


                                                                                  │[N, Ic, ...]
                                                                              ┌───▼───┐
                                                                              │ split │
                                                                              └┬─────┬┘
                                                            ┌──────────────────┘ ... └────────────────┐
           │[N, Ic, ...]                                    │[N, Ic/G, ...]                           │[N, Ic/G, ...]
    ┌──────▼──────┐                                  ┌──────▼──────┐                           ┌──────▼──────┐
    │ convolution ◄──W [Oc, Ic/G, ...]   replace     │ convolution ◄──W [Oc/G, Ic/G, ...]      │ convolution ◄──W [Oc/G, Ic/G, ...]
    │   group=G   ◄──B [Oc]             ────────►    │   group=1   ◄──B [Oc/G]            ...  │   group=1   ◄──B [Oc/G]
    └──────┬──────┘                       with       └──────┬──────┘                           └──────┬──────┘
           ▼[N, Oc, ...]                                    │ [N, Oc/G, ...]                          │[N, Oc/G, ...]
                                                            └──────────────────┐ ... ┌────────────────┘
                                                                              ┌▼─────▼┐
                                                                              │  cat  │
                                                                              └───┬───┘
                                                                                  ▼[N, Oc, ...]
    """

    module: GraphModule

    def _get_tensor_constant_from_node(self, node) -> Parameter | None:
        """Get the static data from a given node. If it doesn't have any data, return `None`."""
        if node is None or node.op != "get_attr":
            return None

        target_atoms = node.target.split(".")
        attr_itr = self.module
        for atom in target_atoms:
            if not hasattr(attr_itr, atom):
                return None
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

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

    def _create_split_node(self, *split_args) -> Node:
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

    def _create_convolution_node(self, conv_target, args: tuple) -> Node:
        convolution_node = self.module.graph.call_function(conv_target, args)

        # Assign the `source_fn_stack` and `val` meta fields as they are required for quantization.
        convolution_node.meta["source_fn_stack"] = [
            (convolution_node.name, torch.convolution)
        ]

        # Compute the output shapes for the `convolution`, and assign the `val` meta.
        with FakeTensorMode() as mode:
            input_shapes = [
                input_.meta["val"].shape if hasattr(input_, "meta") else input_.shape
                for input_ in args[:3]
            ]
            input_dtypes = [
                input_.meta["val"].dtype if hasattr(input_, "meta") else input_.dtype
                for input_ in args[:3]
            ]
            fake_inputs = [
                FakeTensor.from_tensor(torch.empty(shape, dtype=dtype), mode)
                for shape, dtype in zip(input_shapes, input_dtypes)
            ]
            output = conv_target(*fake_inputs, *args[3:])
            convolution_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output.shape, dtype=output.dtype), mode
            )

        return convolution_node

    def _create_concat_node(self, *cat_args) -> Node:
        cat_target = torch.ops.aten.cat.default
        concat_node = self.module.graph.call_function(cat_target, cat_args)

        # Assign the `source_fn_stack` and `val` meta fields as they are required for quantization.
        concat_node.meta["source_fn_stack"] = [(concat_node.name, torch.cat)]

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
            concat_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output.shape, dtype=output.dtype), mode
            )

        return concat_node

    def _get_topologically_last_node(self, nodes: list[Node]) -> Node:
        """Return the node from `nodes` which appears last in the graph."""
        for node in reversed(self.module.graph.nodes):
            if node in nodes:
                return node

        raise RuntimeError(f"None of the nodes `{nodes}` are in the graph.")

    def _create_parameter_node_for_data(
        self, data: torch.Tensor, name: str, insert_after_node: torch.Node
    ) -> torch.Node:
        """Create a parameter node in the graph, which contains the provided `data`."""
        new_name = get_new_attr_name_with_prefix(name)(self.module)

        # Create the node for the parameter.
        param = torch.nn.Parameter(data, False)
        _assign_attr(param, self.module, str(new_name), _AttrKind.PARAMETER)
        with self.module.graph.inserting_after(insert_after_node):
            static_parameter_node = self.module.graph.get_attr(new_name)

        with FakeTensorMode() as mode:
            static_parameter_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(data.shape, dtype=data.dtype), mode
            )

        return static_parameter_node

    def call(self, module: GraphModule):
        self.module = module

        def _is_conv(node_: Node):
            return node_.op == "call_function" and node_.target in (
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
            )

        made_changes = False

        for node in self.module.graph.nodes:
            if not _is_conv(conv_node := node):
                continue

            if len(conv_node.args) < 7:
                # The `aten.conv` can have fewer args if the others use default values.
                #  So in this case, `groups == 1`.
                continue
            x, w, b, stride, padding, dilation, groups = conv_node.args

            if not group_conv_convertible_into_multiple_convolutions(conv_node, groups):
                continue

            if len(x.meta["val"].shape) not in [3, 4]:
                # Only 1D and 2D convolutions are supported by the Neutron backend. Don't decompose anything else.
                continue

            w_data = self._get_tensor_constant_from_node(w)
            b_data = self._get_tensor_constant_from_node(b)
            if w_data is None or b_data is None:
                continue  # Only the standard case with static weights and bias is supported.

            # Create a `split` node to split the main input.
            # Split across dimension `1` (channels), `groups` slices of size `input_split_size`.
            num_input_channels = x.meta["val"].shape[1]
            input_split_sizes = [num_input_channels // groups] * groups
            with self.module.graph.inserting_before(conv_node):
                split_node = self._create_split_node(x, input_split_sizes, 1)

            # Add `GetItem` nodes to extract the outputs of the `split_node`.
            split_getitem_nodes = [
                self._create_and_insert_get_item_node(split_node, i)
                for i in range(groups)
            ]

            # Split the weights and bias, across dimension `0`, slices of size `weight_split_size`.
            weight_split_size = w.meta["val"].shape[0] // groups
            split_weights_data = torch.split(w_data, weight_split_size, 0)
            split_bias_data = torch.split(b_data, weight_split_size, 0)

            # Turn the weights and biases into parameter nodes containing the data.
            # Use a different name for every parameter. The function internally ensures the name's uniqueness, but
            #  relying on it sometimes causes strange failures when `groups > 5` for some weird reason.
            split_weight_nodes = [
                self._create_parameter_node_for_data(
                    weight_data, w.name + f"_{i}_", split_node
                )
                for i, weight_data in enumerate(split_weights_data)
            ]
            split_bias_nodes = [
                self._create_parameter_node_for_data(
                    bias_data, b.name + f"_{i}_", split_node
                )
                for i, bias_data in enumerate(split_bias_data)
            ]

            # Create the `conv` nodes.
            with self.module.graph.inserting_after(
                self._get_topologically_last_node(
                    split_getitem_nodes + split_weight_nodes + split_bias_nodes
                )
            ):
                split_conv_nodes = [
                    self._create_convolution_node(
                        conv_node.target,  # Use the same target as the original convolution (1d/2d/3d/...).
                        (input_getitem, weight, bias, stride, padding, dilation, 1),
                    )
                    for input_getitem, weight, bias in zip(
                        split_getitem_nodes, split_weight_nodes, split_bias_nodes
                    )
                ]

            # Create the `cat` node.
            with self.module.graph.inserting_after(
                self._get_topologically_last_node(split_conv_nodes)
            ):
                concat_node = self._create_concat_node(
                    split_conv_nodes, 1
                )  # Concatenate along the channels.

            # Replace the uses of the original convolution with the `concat_node`.
            conv_node.replace_all_uses_with(concat_node)
            self.module.graph.erase_node(conv_node)

            made_changes = True

        return PassResult(self.module, made_changes)
