# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec

from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class MoveActivationBeforeConcat(PassBase):
    """Move some operators around in the following pattern.
    This is a common pattern that emerges from the conversion of separable convolutions.
    This optimization works together with joint quantization of compute nodes and activations. Without it,
    it is not beneficial.

             │                    │                               │                     │
      ┌──────▼──────┐      ┌──────▼──────┐                 ┌──────▼──────┐       ┌──────▼──────┐
      │ aten.conv2d │  ... │ aten.conv2d │                 │ aten.conv2d │  ...  │ aten.conv2d │
      └──────┬──────┘      └──────┬──────┘                 └──────┬──────┘       └──────┬──────┘
             └───────┐     ┌──────┘                               │                     │
                  ┌──▼─────▼─┐           replace with       ┌─────▼─────┐         ┌─────▼─────┐
                  │ aten.cat │          ──────────────►     │ aten.relu │   ...   │ aten.relu │
                  └────┬─────┘                              └─────┬─────┘         └─────┬─────┘
                       │                                          └───────┐     ┌───────┘
                 ┌─────▼─────┐                                         ┌──▼─────▼─┐
                 │ aten.relu │                                         │ aten.cat │
                 └─────┬─────┘                                         └────┬─────┘
                       │                                                    │
    """

    def __init__(self, neutron_target_spec: NeutronTargetSpec):
        self.neutron_target_spec = neutron_target_spec

    def call(self, module: GraphModule) -> bool:
        def _is_concat(node_: Node) -> bool:
            return (
                node_.op == "call_function"
                and node_.target == torch.ops.aten.cat.default
            )

        made_changes = False

        for node in module.graph.nodes:
            if not _is_concat(node):
                continue  # Not cat node.

            cat_node = node
            activation = next(iter(cat_node.users))

            # Check if all cat inputs nodes are conv 2D or linear 2D type and their only user is cat.
            if not all(
                self.neutron_target_spec.neutron_target_info.is_fusable_conv_or_linear__aten(
                    input_node
                )
                and len(input_node.users) == 1
                for input_node in cat_node.all_input_nodes
            ):
                continue

            # Check if following activation is supported on Neutron as fused activation.
            if not (
                len(cat_node.users) == 1
                and self.neutron_target_spec.neutron_target_info.is_supported_fused_activation__aten(
                    activation
                )
            ):
                continue

            # Loop all Cat input nodes and insert new activation after node.
            for input_node in cat_node.all_input_nodes:
                with module.graph.inserting_after(input_node):
                    new_activation = module.graph.call_function(
                        activation.target,
                        args=(),
                        kwargs=activation.kwargs,
                    )

                    new_activation.meta["source_fn_stack"] = [
                        (
                            new_activation.name,
                            activation.meta["source_fn_stack"][-1][-1],
                        )
                    ]
                    new_activation.meta["val"] = input_node.meta["val"]

                    # Replace the uses of the input node with the new activation node.
                    input_node.replace_all_uses_with(new_activation)
                    new_activation.args = (input_node, *activation.args[1:])

            # Replace the uses of the activation node with the cat node.
            activation.replace_all_uses_with(cat_node)

            module.graph.erase_node(activation)

            made_changes = True

        return PassResult(module, made_changes)
