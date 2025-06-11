# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


def get_group_norm_decomposition(op) -> tuple:
    if op == exir_ops.edge.aten.native_group_norm.default:
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
    if op == torch.ops.aten.group_norm.default:
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
    raise RuntimeError(f"Can't get group_norm composition for op {op}")


class DecomposeGroupNormPass(ArmPass):
    """
    groupnorm is defined as: ((x - E[x]) / sqrt(Var[x] + eps)) * weights + bias
    Decompose groupnorm(x, weight, bias, N, C, HxW, group, eps) to a sequence of:
    mean        = op_mean(x, dims)           # E[x]
    var         = op_var(x, dims)            # Var[x]
    numerator   = op_sub(x, mean)            # (x - E[x])
    add         = op_add(var, eps)           # Var[x] + eps
    rsqrt       = op_rsqrt(add)              # 1 / sqrt(Var[x] + eps)
    mul         = op_mul(numerator, rsqrt)   # ((x - E[x]) / sqrt(Var[x] + eps))
    weigths     = op_mul(mul, weigths)       # ((x - E[x]) / sqrt(Var[x] + eps)) * weigths
    bias        = op_add(weigths, bias)      # ((x - E[x]) / sqrt(Var[x] + eps)) * weigths + bias
    where x can viewed with shape [N, group, C//group, HxW] dims=[C//group, HxW]

    Source: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    """

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in (
                exir_ops.edge.aten.native_group_norm.default,
                torch.ops.aten.group_norm.default,
            ):
                continue

            # epsilon default value
            eps = torch.finfo().eps
            weights = None
            bias = None
            args = node.args
            meta = node.meta
            if isinstance(meta["val"], tuple):
                shape = meta["val"][0].size()
                dtype = meta["val"][0].dtype
            else:
                shape = meta["val"].size()
                dtype = meta["val"].dtype
            match len(args):
                # MI profile always provides all the args: x, weight, bias, N, C, HxW, group, eps
                case 8:
                    x, weights, bias, N, C, HxW, group, eps = args
                # BI profile: affine=[True|False], eps!=1e-5
                case 5:
                    x, group, weights, bias, eps = args
                # BI profile: affine=True, eps=1e-5
                case 4:
                    x, group, weights, bias = args
                # BI profile: affine=False, eps=1e=5
                case 2:
                    x, group = args
                # Unsupported args
                case _:
                    raise ValueError(
                        f"Unsupported group_norm argument pattern with {len(args)} args"
                    )
            N = shape[0]
            C = shape[1]
            HxW = 1
            for dim in shape[2:]:
                HxW *= dim
            channels_per_group = C // group
            grouped_shape = torch.Size([N, group, channels_per_group, HxW])
            dims = [2, 3]
            epsilon_reshaped_shape = torch.Size([1] * len(grouped_shape))
            weights_reshaped_shape = torch.Size([1, group, channels_per_group, 1])
            (
                mean_op,
                sub_op,
                var_op,
                full_op,
                add_op,
                rsqrt_op,
                mul_op,
                view_op,
            ) = get_group_norm_decomposition(node.target)
            with graph_module.graph.inserting_before(node):
                keepdim = True
                x_reshaped = create_node(
                    graph_module.graph,
                    view_op,
                    args=(x, grouped_shape),
                    from_node=node,
                )
                mean = create_node(
                    graph_module.graph, mean_op, args=(x_reshaped, dims, keepdim)
                )
                sub = create_node(graph_module.graph, sub_op, args=(x_reshaped, mean))
                var = create_node(
                    graph_module.graph,
                    var_op,
                    args=(x_reshaped, dims),
                    kwargs={"correction": 0, "keepdim": keepdim},
                    from_node=node,
                )
                full = create_node(
                    graph_module.graph,
                    full_op,
                    args=(epsilon_reshaped_shape, eps),
                    kwargs={"dtype": dtype},
                    from_node=node,
                )
                add0 = create_node(
                    graph_module.graph, add_op, args=(var, full), from_node=node
                )
                rsqrt = create_node(
                    graph_module.graph, rsqrt_op, args=(add0,), from_node=node
                )
                mul0 = create_node(
                    graph_module.graph, mul_op, args=(sub, rsqrt), from_node=node
                )
                if weights is not None:
                    weights_reshaped = create_node(
                        graph_module.graph,
                        view_op,
                        args=(weights, weights_reshaped_shape),
                        from_node=node,
                    )
                    mul1 = create_node(
                        graph_module.graph,
                        mul_op,
                        args=(
                            mul0,
                            weights_reshaped,
                        ),
                        from_node=node,
                    )
                else:
                    mul1 = mul0
                if bias is not None:
                    bias_reshaped_shape = weights_reshaped_shape
                    bias_reshaped = create_node(
                        graph_module.graph,
                        view_op,
                        args=(bias, bias_reshaped_shape),
                        from_node=node,
                    )
                    output = create_node(
                        graph_module.graph,
                        add_op,
                        args=(mul1, bias_reshaped),
                        from_node=node,
                    )
                else:
                    output = mul1

                output_reshaped = create_node(
                    graph_module.graph,
                    view_op,
                    args=(output, shape),
                    from_node=node,
                )

                users = [user for user in node.users if node != user]
                node.replace_all_uses_with(output_reshaped)
                for user in users:
                    if user.target == operator.getitem:
                        user.replace_all_uses_with(output_reshaped)
                graph_module.graph.erase_node(node)
                graph_module.graph.eliminate_dead_code()
                modified = True
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
