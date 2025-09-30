# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import prod
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.add_bias_pass import AddBiasPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.quant_args import QuantArgs

from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class DecomposeCumsumPass(ArmPass):
    """
    Decomposes cumsum into a 1D convolution with a kernel of ones.

    For example, the cumsum of an input tensor [1, 1] is [1, 1 + 1] = [1, 2].
    To decompose this, take the input tensor and pre-padded with len(input)-1 zeros and
    slided over with a kernel [1,1], of length len(input):

    Input:  [0, 1, 1]
    Kernel: [1, 1]       = [1]
               [1, 1]    = [2]

    Since pytorch only supports symmetric padding, in reality the result will have
    an additional 1 calculated at the end, which leads to an required extra slice op.

    To extend this to higher dimensions, the input is reshaped to [N, C, H, W] with
       N = <dims before cumsum dim>
       C = 1
       H = <cumsum dim>
       W = <dims after cumsum dim>
    And the convolution is applied over dimension H.
    """

    _passes_required_after: Set[Type[ExportPass]] = {AddBiasPass}

    def call(self, graph_module):
        graph = graph_module.graph
        targets = (exir_ops.edge.aten.cumsum.default, torch.ops.aten.cumsum.default)
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in targets:
                continue

            if len(node.args) != 2:
                raise ValueError(
                    "Cumsum node should have exactly two arguments: input and dim."
                )

            # Get node data
            input_node, dim = node.args
            val = node.meta.get("val")
            original_shape = list(val.shape)
            dtype = input_node.meta.get("val").dtype
            dim = dim % len(original_shape)

            # Compute shapes
            pre_cumsum_dim = prod(original_shape[:dim]) if dim > 0 else 1
            cumsum_dim = original_shape[dim]
            post_cumsum_dim = (
                prod(original_shape[dim + 1 :]) if dim < len(original_shape) - 1 else 1
            )
            conv_shape = [
                pre_cumsum_dim,
                1,
                cumsum_dim,
                post_cumsum_dim,
            ]
            pad_shape = [original_shape[dim] - 1, 0]
            weight_shape = [1, 1, original_shape[dim], 1]

            # Create convolution weight
            with graph.inserting_before(list(graph.nodes)[0]):
                weight_data = torch.ones(size=weight_shape, dtype=dtype)
                weight_node = create_constant_placeholder(
                    self.exported_program,
                    graph,
                    node.name + "_kernel",
                    InputKind.PARAMETER,
                    weight_data,
                )

            # Create decomposed nodes
            view_op = exir_ops.edge.aten.view_copy.default
            conv_op = exir_ops.edge.aten.convolution.default
            slice_op = exir_ops.edge.aten.slice_copy.Tensor
            with graph.inserting_before(node):
                # Reshape to 4D with
                view_args = (input_node, conv_shape)
                view_node = create_node(graph, view_op, args=view_args, from_node=node)

                conv_args = (
                    view_node,
                    weight_node,
                    None,
                    [1, 1],
                    pad_shape,
                    [1, 1],
                    False,
                    [0],
                    1,
                )
                conv_node = create_node(graph, conv_op, args=conv_args, from_node=node)

                # The convolution is inserted after quantization, so we need to set our
                # own quantization parameters for the weights here. However since the
                # data is ones directly created as int8, they already have correct scale
                # and so no scaling needs to be done, i.e. set scale=1.0, zero_point=0.0
                if (
                    "input_qparams" in conv_node.meta
                    and len(conv_node.meta["input_qparams"]) > 0
                ):
                    qparams = QuantArgs(1.0, 0.0, -128, 127, torch.int8)
                    conv_node.meta["input_qparams"][1] = qparams

                slice_args = (conv_node, 2, 0, original_shape[dim])
                slice_node = create_node(
                    graph, slice_op, args=slice_args, from_node=node
                )

                view_original_args = (slice_node, original_shape)
                view_original_node = create_node(
                    graph, view_op, args=view_original_args, from_node=node
                )

            # Replace and remove original
            node.replace_all_uses_with(view_original_node)
            graph.erase_node(node)
            modified = True

        if modified:
            # Cleanup
            graph.eliminate_dead_code()
            graph_module.recompile()
            # Apply any operator-level transforms
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
