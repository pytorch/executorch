# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numpy as np
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm.tosa_quant_utils import dq_op, q_op
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeLinearPass(ExportPass):
    """
    This pass decomposes linear into a Conv2D with the required view operations.
    linear(x, weights, bias) becomes:
        x_reshaped       = view(x)
        weights_reshaped = view(weights)
        conv2d           = conv2d(x_reshaped, weights_reshaped, bias)
        output           = view(conv2d)
    It also inserts q/dq pairs if the linear node was quantized.
    """

    def call(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.linear.default:
                continue
            args = node.args
            input = args[0]
            weights = args[1]
            bias = args[2] if len(args) > 2 else None
            output_shape = get_first_fake_tensor(node).shape
            input_shape = get_first_fake_tensor(input).shape
            weights_shape = get_first_fake_tensor(weights).shape
            batches = int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
            # input has shape (..., Ci)
            input_reshaped_shape = [batches, input_shape[-1], 1, 1]
            # weights have shape (Co, Ci)
            weights_reshaped_shape = [weights_shape[0], weights_shape[1], 1, 1]

            with graph_module.graph.inserting_before(node):
                quantize = input.op == "call_function" and input.target == dq_op
                q_params = input.args[1:] if quantize else None
                # Reshape input to 4D with shape (N, Ci, 1, 1)
                input_reshaped = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.view_copy.default,
                    args=(input, input_reshaped_shape),
                    kwargs={},
                    quantize=quantize,
                    q_params=q_params,
                )

                quantize = weights.op == "call_function" and weights.target == dq_op
                q_params = weights.args[1:] if quantize else None
                # Reshape weights to 4D with shape (Co, Ci, 1, 1)
                weights_reshaped = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.view_copy.default,
                    args=(weights, weights_reshaped_shape),
                    kwargs={},
                    quantize=quantize,
                    q_params=q_params,
                )

                consumer_node = list(node.users)[0]
                quantize = (
                    consumer_node.op == "call_function" and consumer_node.target == q_op
                )
                q_params = consumer_node.args[1:] if quantize else None
                conv = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.convolution.default,
                    args=(
                        input_reshaped,
                        weights_reshaped,
                        bias,
                        [1, 1],  # strides
                        [0, 0],  # padding
                        [1, 1],  # dilation
                        False,  # transposed
                        [0, 0],  # output padding
                        1,  # groups
                    ),
                    kwargs={},
                    quantize=quantize,
                    q_params=q_params,
                )

            with graph_module.graph.inserting_after(conv):
                # Reshape output to same rank as original input with shape (..., Co)
                # No need to insert q/dq pair as Conv2D node above has inserted them if
                # required.
                output = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.view_copy.default,
                    args=(conv, list(output_shape)),
                    kwargs={},
                )

            node.replace_all_uses_with(output)
            graph_module.graph.erase_node(node)
            graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
