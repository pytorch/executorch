# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    insert_q_dq_pair,
)
from executorch.backends.arm.tosa_quant_utils import dq_op, q_op
from executorch.backends.arm.tosa_utils import is_consumer_node_depthwise_conv2d
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.library import impl, Library

# Define lib with passthrough operators. The operators have no real meaning in edge IR
# except for argument validaiton and a passthrough output. The operators will be used
# when lowering to TOSA, e.g. a passthrough_to_tosa._transpose will not affect
# the edge IR graph but will be lowered to a TOSA-TRANSPOSE.
lib = Library("passthrough_to_tosa", "DEF")
# For certain operators we need the data in a specific data format. Changing tosa_dim_order
# is not sufficient as we also need transpose the data.
# By utilizing an edge IR passthrough operator we can keep the edge program in
# channels-first/contiguous and get the desired behavior in the TOSA lowering.
lib.define("_transpose(Tensor self, int[] dim_order) -> Tensor")


@impl(lib, "_transpose")
def _transpose_impl(*args, **kwargs):
    # Validate length of dim_order array
    dim = args[1]
    if len(dim) != 4 and len(dim) != 5:
        raise ValueError(
            f"Dim order length must be either 4 or 5, got {len(dim)}: {dim}"
        )
    # Pass-through in edge-IR
    return args[0]


class AnnotateChannelsLastDimOrder(ExportPass):
    """
    Annotates each node with a tosa_dim_order. tosa_dim_order can be seen as a channels-last dim-order
    that in most cases will be (0, 2, 3, 1) for nodes with 4D-shapes. The pass also inserts passthrough_to_tosa._transpose
    when a transition between 3D and 4D/5D tensors happen.
    The annotated tosa_dim_order is used to permute the node's shape such that it gives a TOSA-compliant shape.
    """

    NHWC_order = (0, 2, 3, 1)
    NHWC_inverse_order = (0, 3, 1, 2)
    HWCM_order = (2, 3, 0, 1)
    NNHWC_order = (0, 1, 3, 4, 2)
    NNHWC_inverse_order = (0, 1, 4, 2, 3)

    def is_weight_node_for_depthwise_conv2d(self, node: torch.fx.Node):
        """
        returns True for dq and w in the following sequences;
        w -> depthwise_conv2d -> ...
        w -> dq -> depthwise_conv2d -> ...
        """
        if node.op == "call_function":
            if node.target != dq_op:
                return False
            prev_node = node.args[0]
            if cast(torch.fx.Node, prev_node).op != "placeholder":
                return False
            if is_consumer_node_depthwise_conv2d(node):
                consumer_node = list(node.users)[0]
                return consumer_node.args[1] == node
        elif node.op == "placeholder":
            # node is an input, weight or bias node
            consumer_node = list(node.users)[0]
            if self.is_weight_node_for_depthwise_conv2d(consumer_node):
                return True
            if is_consumer_node_depthwise_conv2d(node):
                # Check that node is the weight-argument and not input or bias
                return consumer_node.args[1] == node

        return False

    @staticmethod
    def memory_format_differs(shape):
        """Returns true if the shape will have a different memory layout in (N)NCHW and (N)NHWC format"""
        if len(shape) >= 5:
            C = shape[2]
            H = shape[3]
            W = shape[4]
        elif len(shape) == 4:
            C = shape[1]
            H = shape[2]
            W = shape[3]
        elif len(shape) == 3:
            C = shape[0]
            H = shape[1]
            W = shape[2]
        if len(shape) <= 2:
            return False

        return C > 1 and (H > 1 or W > 1)

    @staticmethod
    def is_channel_reshape(input_shape, output_shape):
        """Returns true if the reshape changes the channel dimension"""
        if not (
            (len(input_shape) == len(output_shape) and (len(output_shape) in (4, 5)))
            or (len(input_shape) == 4 and len(output_shape) == 5)
            or (len(input_shape) == 5 and len(output_shape) == 4)
        ):
            return False

        C_old = input_shape[-3]
        C_new = output_shape[-3]

        N_new = (
            output_shape[0]
            if len(output_shape) == 4
            else output_shape[0] * output_shape[1]
        )
        N_old = (
            input_shape[0] if len(input_shape) == 4 else input_shape[0] * input_shape[1]
        )

        return (N_old != N_new) or (C_old != C_new)

    @staticmethod
    def insert_input_transpose(node, input_node, graph_module):
        quantize = input_node.target == dq_op
        q_params = input_node.args[1:] if quantize else None
        with graph_module.graph.inserting_before(node):
            permute_node = create_node(
                graph_module.graph,
                torch.ops.passthrough_to_tosa._transpose.default,
                args=(
                    input_node,
                    list(
                        AnnotateChannelsLastDimOrder.NNHWC_inverse_order
                        if len(get_first_fake_tensor(input_node).size()) == 5
                        else AnnotateChannelsLastDimOrder.NHWC_inverse_order
                    ),
                ),
                quantize=quantize,
                q_params=q_params,
            )
            node.replace_input_with(input_node, permute_node)

            permute_node.meta["tosa_dim_order"] = tuple(
                range(len(input_node.meta["val"].size()))
            )
            permute_node.meta["val"] = input_node.meta["val"]

    @staticmethod
    def insert_output_transpose(node, graph_module):
        with graph_module.graph.inserting_after(node):
            permute_node = create_node(
                graph_module.graph,
                torch.ops.passthrough_to_tosa._transpose.default,
                args=(
                    node,
                    list(
                        AnnotateChannelsLastDimOrder.NNHWC_order
                        if len(get_first_fake_tensor(node).size()) == 5
                        else AnnotateChannelsLastDimOrder.NHWC_order
                    ),
                ),
            )
            permute_node.meta["tosa_dim_order"] = (
                AnnotateChannelsLastDimOrder.NNHWC_order
                if len(get_first_fake_tensor(node).size()) == 5
                else AnnotateChannelsLastDimOrder.NHWC_order
            )
            permute_node.meta["val"] = get_first_fake_tensor(node).permute(
                AnnotateChannelsLastDimOrder.NNHWC_order
                if len(get_first_fake_tensor(node).size()) == 5
                else AnnotateChannelsLastDimOrder.NHWC_order
            )
            node.meta["tosa_dim_order"] = tuple(
                range(len(get_first_fake_tensor(node).size()))
            )
            users = [user for user in node.users if user != permute_node]
            for user in users:
                user.replace_input_with(node, permute_node)

            quantize = node.args[0] == q_op
            if quantize:
                q_params = node.args[0].args[1:]
                insert_q_dq_pair(graph_module.graph, node, q_params)

    @staticmethod
    def _insert_view_transpose(
        input_shape, output_shape, node, input_node, graph_module
    ):
        nchw_to_nhwc = len(input_shape) < 4 and len(output_shape) >= 4
        nhwc_to_nchw = len(input_shape) >= 4 and len(output_shape) < 4
        channel_reshape = AnnotateChannelsLastDimOrder.is_channel_reshape(
            output_shape, input_shape
        )

        if (
            channel_reshape or nhwc_to_nchw
        ) and AnnotateChannelsLastDimOrder.memory_format_differs(input_shape):
            AnnotateChannelsLastDimOrder.insert_input_transpose(
                node, input_node, graph_module
            )
        if (
            channel_reshape or nchw_to_nhwc
        ) and AnnotateChannelsLastDimOrder.memory_format_differs(output_shape):
            AnnotateChannelsLastDimOrder.insert_output_transpose(node, graph_module)

    def insert_tosa_transposes(self, graph_module: torch.fx.GraphModule):
        """
        Transposes are needed for operators transforming the input to a different rank, as 4D and 5D-tensors are assumed to be in (N)NHWC-format, whereas all other are in (N)NCHW format.
        This is relevant for the following cases:
        - view:       <4D ->  >=4D
        - view:      >=4D ->   <4D
        Additionally, a 4D/5D->4D/5D view operation acting on the channel dimension currently needs to be performed in (N)NCHW format, leadning to one extra input and output transpose for this case.

        Transposes can be avoided for shapes where there is no difference in actual memory, e.g for
        - H == W == 1
        - C == 1
        - 1D/2D tensors
        """
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            elif node.target == exir_ops.edge.aten.view_copy.default:
                input_node = node.args[0]
                input_shape = input_node.meta["val"].shape
                output_shape = node.meta["val"].shape

                self._insert_view_transpose(
                    input_shape, output_shape, node, input_node, graph_module
                )

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            node_data = get_first_fake_tensor(node).data

            if node_data.dim() == 4:
                dim_order = self.NHWC_order
                if self.is_weight_node_for_depthwise_conv2d(node):
                    # The weights of TOSA DEPTHWISE_CONV2D have shape (H, W, C, M) which corresponds to
                    # dim_order = (2, 3, 0, 1) (https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d).
                    dim_order = self.HWCM_order
            elif node_data.dim() == 5:
                dim_order = self.NNHWC_order  # type: ignore[assignment]
            else:
                dim_order = tuple(range(node_data.dim()))  # type: ignore[assignment]
            node.meta["tosa_dim_order"] = dim_order
        # Insert TOSA transposes to convert between (N)NCHW and (N)NHWC format.
        # See insert_tosa_transposes for insertion conditions.
        self.insert_tosa_transposes(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
