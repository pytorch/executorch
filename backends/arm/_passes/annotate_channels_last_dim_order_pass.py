# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
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
from executorch.backends.arm.tosa_quant_utils import dq_op, q_op, register_passable_op
from executorch.backends.arm.tosa_utils import is_consumer_node_depthwise_conv2d
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.library import impl, Library

# Define lib with passthrough operators. The operators have no real meaning in edge IR
# except for argument validaiton and a passthrough output. The operators will be used
# when lowering to TOSA, e.g. a passthrough_to_tosa._transpose will not affect
# the edge IR graph but will be lowered to a TOSA-TRANSPOSE.
lib = Library("passthrough_to_tosa", "DEF")
# For operators that change the rank of the input, such as unsqueeze and squeeze, we may need
# to switch dim_order before the opertation. Changing tosa_dim_order is not sufficient
# as we also need transpose the data into the correct data format.
# By utilizing an edge IR passthrough operator we can keep the edge program in
# channels-first/contiguous and get the desired behavior in the TOSA lowering.
lib.define("_transpose(Tensor self, int[] dim_order) -> Tensor")


@impl(lib, "_transpose")
def _transpose_impl(*args, **kwargs):
    # Validate length of dim_order array
    dim = args[1]
    assert len(dim) <= 4
    # Pass-through in edge-IR
    return args[0]


register_passable_op(torch.ops.passthrough_to_tosa._transpose)


class AnnotateChannelsLastDimOrder(ExportPass):
    """
    Annotates each node with a tosa_dim_order. tosa_dim_order can be seen as a channels-last dim-order
    that in most cases will be (0, 2, 3, 1) for nodes with 4D-shapes. The pass also inserts passthrough_to_tosa._transpose
    when a transition between 3D and 4D tensors happen.
    The annotated tosa_dim_order is used to permute the node's shape such that it gives a TOSA-compliant shape.
    """

    NHWC_order = (0, 2, 3, 1)
    NHWC_inverse_order = (0, 3, 1, 2)
    HWCM_order = (2, 3, 0, 1)

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

    def insert_input_transpose(self, node, input_node, graph_module):
        quantize = input_node.target == dq_op
        q_params = input_node.args[1:] if quantize else None
        with graph_module.graph.inserting_before(node):
            permute_node = create_node(
                graph_module.graph,
                torch.ops.passthrough_to_tosa._transpose,
                args=(input_node, list(self.NHWC_inverse_order)),
                quantize=quantize,
                q_params=q_params,
            )
            node.replace_input_with(input_node, permute_node)

            permute_node.meta["tosa_dim_order"] = tuple(
                range(len(input_node.meta["val"].size()))
            )

    def insert_output_transpose(self, node, graph_module):
        with graph_module.graph.inserting_after(node):
            permute_node = create_node(
                graph_module.graph,
                torch.ops.passthrough_to_tosa._transpose,
                args=(node, list(self.NHWC_order)),
            )
            permute_node.meta["tosa_dim_order"] = self.NHWC_order
            node.meta["tosa_dim_order"] = (0, 1, 2, 3)
            users = [user for user in node.users if user != permute_node]
            for user in users:
                user.replace_input_with(node, permute_node)

            quantize = node.args[0] == q_op
            if quantize:
                q_params = node.args[0].args[1:]
                insert_q_dq_pair(graph_module.graph, node, q_params)

    def insert_tosa_transposes(self, graph_module: torch.fx.GraphModule):
        """
        Reshape operations are not equivalent in NCHW and NHWC.
        To get around this, transposes need to be added if the previous or new shape
        fulfil the following condition:
            C > 1 and (H or W > 1)

        This is relevant for the following operations;
        squeeze:     4D ->  3D
        unsqueeze:  <4D ->  4D
        view:       <4D ->  4D
        view:        4D -> <4D
        view:        4D ->  4D
        """

        def transpose_condition(shape):
            if len(shape) != 4:
                return False
            C = shape[1]
            H = shape[2]
            W = shape[3]
            return C > 1 and (H > 1 or W > 1)

        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == exir_ops.edge.aten.squeeze_copy.dims:
                input_node = node.args[0]
                input_shape = input_node.meta["val"].shape
                if transpose_condition(input_shape):
                    self.insert_input_transpose(node, input_node, graph_module)

            elif node.target == exir_ops.edge.aten.unsqueeze_copy.default:
                output_shape = node.meta["val"].shape
                if transpose_condition(output_shape):
                    self.insert_output_transpose(node, graph_module)

            elif node.target == exir_ops.edge.aten.view_copy.default:
                input_node = node.args[0]

                old_shape = input_node.meta["val"].shape
                new_shape = node.meta["val"].shape

                if transpose_condition(old_shape):
                    self.insert_input_transpose(node, input_node, graph_module)

                if transpose_condition(new_shape):
                    self.insert_output_transpose(node, graph_module)

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            node_data = get_first_fake_tensor(node).data

            if node_data.dim() == 4:
                dim_order = self.NHWC_order
                if self.is_weight_node_for_depthwise_conv2d(node):
                    # The weights of TOSA DEPTHWISE_CONV2D have shape (H, W, C, M) which corresponds to
                    # dim_order = (2, 3, 0, 1) (https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d).
                    dim_order = self.HWCM_order
            else:
                dim_order = tuple(range(node_data.dim()))
            node.meta["tosa_dim_order"] = dim_order
        # Take care of cases when:
        # 4D (NHWC) -> >4D (NCH)
        # 3D (NCH)  ->  4D (NHWC)
        self.insert_tosa_transposes(graph_module)
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
