# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def get_layer_norm_decomposition(op) -> tuple:
    if op == exir_ops.edge.aten.native_layer_norm.default:
        return (
            exir_ops.edge.aten.mean.dim,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.var.correction,
            exir_ops.edge.aten.full.default,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.rsqrt.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.view_copy.default,
        )
    if op == torch.ops.aten.layer_norm.default:
        return (
            torch.ops.aten.mean.dim,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.var.correction,
            torch.ops.aten.full.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.rsqrt.default,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.view_copy.default,
        )
    raise RuntimeError(f"Can't get layer_norm composition for op {op}")


class DecomposeLayerNormPass(ExportPass):
    """
    layernorm is defined as: ((x - E[x]) / sqrt(Var[x] + eps)) * weights + bias
    Decompose layernorm(x, normalized_shape, weights, bias, eps) to a sequence of:
    mean        = op_mean(x, dims)           # E[x]
    var         = op_var(x, dims)            # Var[x]
    denominator = op_sub(x, mean)            # (x - E[x])
    add         = op_add(var, eps)           # Var[x] + eps
    rsqrt       = op_rsqrt(add)              # 1 / sqrt(Var[x] + eps)
    mul         = op_mul(denominator, rsqrt) # ((x - E[x]) / sqrt(Var[x] + eps)) * weigths
    bias        = op_add(mul, bias)          # ((x - E[x]) / sqrt(Var[x] + eps)) * weigths + bias

    Source: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in (
                exir_ops.edge.aten.native_layer_norm.default,
                torch.ops.aten.layer_norm.default,
            ):
                continue

            # epsilon default value
            epsilon = torch.finfo().eps
            weights = None
            bias = None
            args = node.args
            meta = node.meta
            match len(args):
                case 5:
                    x, normalized_shape, weights, bias, epsilon = args
                case 4:
                    x, normalized_shape, weights, bias = args
                case 3:
                    x, normalized_shape, weights = args
                case _:
                    x, normalized_shape = args

            n_dims = len(normalized_shape)
            if isinstance(meta["val"], tuple):
                shape = meta["val"][0].size()
            else:
                shape = meta["val"].size()
            dtype = meta["val"][0].dtype
            rank = len(shape)
            dims = list(range(-1, -1 * (n_dims + 1), -1))
            dims = [dim % rank for dim in dims]
            weights_reshaped_shape = [shape[i] if i in dims else 1 for i in range(rank)]
            epsilon_reshaped_shape = [1] * rank

            (
                mean_op,
                sub_op,
                var_op,
                full_op,
                add_op,
                rsqrt_op,
                mul_op,
                view_op,
            ) = get_layer_norm_decomposition(node.target)
            with graph_module.graph.inserting_before(node):
                keepdim = True
                mean = create_node(graph_module.graph, mean_op, args=(x, dims, keepdim))
                sub = create_node(graph_module.graph, sub_op, args=(x, mean))
                var = create_node(
                    graph_module.graph,
                    var_op,
                    args=(x, dims),
                    kwargs={"correction": 0, "keepdim": keepdim},
                )
                full = create_node(
                    graph_module.graph,
                    full_op,
                    args=(epsilon_reshaped_shape, epsilon),
                    kwargs={"dtype": dtype},
                )
                add0 = create_node(graph_module.graph, add_op, args=(var, full))
                rsqrt = create_node(graph_module.graph, rsqrt_op, args=(add0,))
                mul0 = create_node(graph_module.graph, mul_op, args=(sub, rsqrt))
                if weights is not None:
                    weights_reshaped = create_node(
                        graph_module.graph,
                        view_op,
                        args=(weights, weights_reshaped_shape),
                    )
                    mul1 = create_node(
                        graph_module.graph, mul_op, args=(mul0, weights_reshaped)
                    )
                else:
                    mul1 = mul0
                output = mul1
                if bias is not None:
                    bias_reshaped_shape = weights_reshaped_shape
                    bias_reshaped = create_node(
                        graph_module.graph, view_op, args=(bias, bias_reshaped_shape)
                    )
                    output = create_node(
                        graph_module.graph, add_op, args=(mul1, bias_reshaped)
                    )

                users = [user for user in node.users if node != user]
                node.replace_all_uses_with(output)
                for user in users:
                    if user.target == operator.getitem:
                        user.replace_all_uses_with(output)
                graph_module.graph.erase_node(node)
                graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
