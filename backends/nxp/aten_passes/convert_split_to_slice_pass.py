# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.nn.parameter import Parameter
from torch.nn.utils import fuse_conv_bn_weights


class ConvertSplitToSlicePass(PassBase):
    def call(self, graph_module: GraphModule) -> Optional[PassResult]:
        def _is_split(node_: Node) -> bool:
            return (
                node_.op == "call_function"
                and node_.target == torch.ops.aten.split_with_sizes
            )

        made_changes = False

        if not any(map(_is_split, graph_module.graph.nodes)):
            return PassResult(
                graph_module, made_changes
            )  # No split nodes in the model.

        for node in graph_module.graph.nodes:
            if not _is_split(node):
                continue  # Not Split node.

            split_node = node

            input_tensor = self._get_tensor_constant_from_node(graph_module, split_node.args[0])
            split_size_or_section = self._get_tensor_constant_from_node(graph_module, split_node.args[1])
            dim = self._get_tensor_constant_from_node(graph_module, split_node.args[2])

            if any(
                t is None for t in (conv_w, bn_rm, bn_rv)
            ):  # The other inputs can be None.
                continue  # The data is not static. Leave this BatchNorm as is (probably a rare case).
            fused_weight, fused_bias = fuse_conv_bn_weights(
                conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b
            )

            # Update the weight and bias for Conv.
            conv_args = list(conv_node.args)
            if len(conv_args) == 2:
                # Fill in the default bias argument.
                conv_args.append(None)

            weight_attr_name = conv_weight_node.target
            _assign_attr(
                fused_weight, graph_module, weight_attr_name, _AttrKind.PARAMETER
            )

            if conv_bias_node is not None:
                bias_attr_name = conv_bias_node.target
                _assign_attr(
                    fused_bias, graph_module, str(bias_attr_name), _AttrKind.PARAMETER
                )
            else:
                # The Conv doesn't have a bias. Create a new one.
                bias_attr_name = weight_attr_name + "_bias"
                _assign_attr(
                    fused_bias, graph_module, bias_attr_name, _AttrKind.PARAMETER
                )
                with graph_module.graph.inserting_before(conv_node):
                    get_bias_node = graph_module.graph.get_attr(bias_attr_name)

                conv_args[2] = get_bias_node

            conv_node.args = tuple(conv_args)

            # Replace the uses of the BatchNorm with the Conv.
            bn_node.replace_all_uses_with(conv_node)

            made_changes = True

        return PassResult(graph_module, made_changes)
