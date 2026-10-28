# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm._passes.insert_rescales_pass import InsertRescaleInt32Pass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeLinearPass(ArmOpTargetedPass):
    """Decompose linear into a 1x1 Conv2D with projection rows in width.

    Symbolic row counts must be nonzero at runtime, as required by TOSA.

    Example:
        linear(x, weights, bias) becomes:
            x_reshaped = permute(view(x, [1, 1, rows, Ci]), [0, 3, 1, 2])
            weights_reshaped = view(weights)
            conv2d = conv2d(x_reshaped, weights_reshaped, bias)
            output = view(permute(conv2d, [0, 2, 3, 1]), output_shape)

    """

    _passes_required_after: Set[Type[ExportPass]] = {InsertRescaleInt32Pass}
    target_ops = (exir_ops.edge.aten.linear.default,)

    def call(self, graph_module):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.linear.default:
                continue
            modified = True
            args = node.args
            input = args[0]
            weights = args[1]
            bias = args[2] if len(args) > 2 else None
            output_shape = get_first_fake_tensor(node).shape
            input_shape = get_first_fake_tensor(input).shape
            weights_shape = get_first_fake_tensor(weights).shape
            rows = torch.sym_int(1)
            for dim in input_shape[:-1]:
                rows *= dim
            # input has shape (..., Ci)
            input_reshaped_shape = [1, 1, rows, input_shape[-1]]
            # weights have shape (Co, Ci)
            weights_reshaped_shape = [weights_shape[0], weights_shape[1], 1, 1]

            # Materialize SymInts into FX nodes; one call shares
            # the symbol subgraph across all three shape lists.
            with graph_module.graph.inserting_before(node):
                n_in, n_w = (
                    len(input_reshaped_shape),
                    len(weights_reshaped_shape),
                )
                all_sizes = graph_module.graph.materialize_symints(
                    [
                        *input_reshaped_shape,
                        *weights_reshaped_shape,
                        *output_shape,
                    ]
                )
                input_reshaped_shape = all_sizes[:n_in]
                weights_reshaped_shape = all_sizes[n_in : n_in + n_w]
                output_size = all_sizes[n_in + n_w :]
                # Keep features innermost until the permutation to NCHW.
                input_reshaped = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.view_copy.default,
                    args=(input, input_reshaped_shape),
                    kwargs={},
                    from_node=node,
                    inherit_qparams=False,
                )
                input_reshaped = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.permute_copy.default,
                    args=(input_reshaped, [0, 3, 1, 2]),
                    from_node=node,
                    inherit_qparams=False,
                )

                # Reshape weights to 4D with shape (Co, Ci, 1, 1)
                weights_reshaped = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.view_copy.default,
                    args=(weights, weights_reshaped_shape),
                    kwargs={},
                    from_node=node,
                    inherit_qparams=False,
                )

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
                    from_node=node,
                    inherit_qparams=True,
                )

                # Inserting before linear keeps conv -> permute -> view in order.
                output = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.permute_copy.default,
                    args=(conv, [0, 2, 3, 1]),
                    from_node=node,
                    inherit_qparams=False,
                )
                # Restore (..., Co); only the convolution inherits linear qparams.
                output = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.view_copy.default,
                    args=(output, output_size),
                    kwargs={},
                    from_node=node,
                    inherit_qparams=False,
                )

            node.replace_all_uses_with(output)
            graph_module.graph.erase_node(node)
            graph_module.graph.eliminate_dead_code()
        if modified:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
