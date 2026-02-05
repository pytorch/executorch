# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeLinearPass(ExportPass):
    """
    Decompose aten.linear into matmul + add to avoid addmm.

    For 2D inputs, we unsqueeze to 3D before decomposition to force the matmul
    code path instead of addmm. The C++ implementation of aten.linear directly
    calls addmm for 2D inputs with bias, which would require implementing
    aoti_torch_mps_addmm_out. By unsqueezing to 3D, we force the matmul path,
    then squeeze back to 2D.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph

        for node in graph.nodes:
            # Check if this is a linear operation
            is_linear = False

            if node.op == "call_function":
                # Match both edge dialect and core aten linear operators
                if node.target == exir_ops.edge.aten.linear.default:
                    is_linear = True
                elif node.target == torch.ops.aten.linear.default:
                    is_linear = True

            if is_linear:
                # Get input, weight, and bias arguments
                input_node = node.args[0]
                weight_node = node.args[1]
                bias_node = node.args[2] if len(node.args) > 2 else None

                with graph.inserting_before(node):
                    # Determine which ops to use based on the input operator
                    target_str = str(node.target)

                    if "executorch_exir_dialects_edge" in target_str:
                        # Use edge dialect operators
                        t_op = exir_ops.edge.aten.t.default
                        matmul_op = exir_ops.edge.aten.matmul.default
                        add_op = exir_ops.edge.aten.add.Tensor
                        unsqueeze_op = exir_ops.edge.aten.unsqueeze.default
                        squeeze_op = exir_ops.edge.aten.squeeze.dims
                    else:
                        # Use core aten operators
                        t_op = torch.ops.aten.t.default
                        matmul_op = torch.ops.aten.matmul.default
                        add_op = torch.ops.aten.add.Tensor
                        unsqueeze_op = torch.ops.aten.unsqueeze.default
                        squeeze_op = torch.ops.aten.squeeze.dims

                    # Check if input is 2D
                    needs_unsqueeze = False
                    if hasattr(input_node, "meta") and "val" in input_node.meta:
                        if len(input_node.meta["val"].shape) == 2:
                            needs_unsqueeze = True

                    # Unsqueeze 2D input to 3D: (M, K) -> (1, M, K)
                    current_input = input_node
                    if needs_unsqueeze:
                        current_input = graph.call_function(
                            unsqueeze_op,
                            args=(input_node, 0),
                        )

                    # Decompose linear: matmul(input, weight.T) + bias
                    weight_t = graph.call_function(
                        t_op,
                        args=(weight_node,),
                    )

                    matmul_result = graph.call_function(
                        matmul_op,
                        args=(current_input, weight_t),
                    )

                    if bias_node is not None:
                        result = graph.call_function(
                            add_op,
                            args=(matmul_result, bias_node),
                        )
                    else:
                        result = matmul_result

                    # Squeeze 3D output back to 2D: (1, M, N) -> (M, N)
                    if needs_unsqueeze:
                        result = graph.call_function(
                            squeeze_op,
                            args=(result, [0]),
                        )

                # Replace all uses of the linear node with the decomposed result
                node.replace_all_uses_with(result)
                graph.erase_node(node)
                modified = True

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
