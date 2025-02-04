# Copyright 2025 Arm Limited and/or its affiliates.
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


edge_bn_ops = (exir_ops.edge.aten._native_batch_norm_legit_no_training.default,)


def get_bn_decomposition(op) -> tuple:
    """
    Returns decomposition of batchnorm in edge ops.
    Raises RuntimeError if op is not batchnorm edge op.
    """
    if op in edge_bn_ops:
        return (
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.rsqrt.default,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.full.default,
        )
    else:
        raise RuntimeError(f"Can't get decomposition for {op}")


class DecomposeBatchNormPass(ExportPass):
    """
    Decompose BatchNorm to:
    %output = (%x - %E[x]) /  SQRT( %Var[x] + %epsilon ) * %gamma + %beta
    e.g.
    %output = (%activations - %running_mean) /  SQRT( %running_var + %epsilon_const ) * %weights +  %bias
    ->
    %op1 = sub(%activations, %running_mean)
    %op2 = add(%running_var, %epsilon_const)
    %op3 = rsqrt(%op2)
    %op4 = mul(%op1, %op3)
    %op5 = mul(%op4, %weights)
    %output = add(%op5, %bias)
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in edge_bn_ops:
                continue

            args = node.args
            meta = node.meta
            (
                activations,
                weights,
                bias,
                running_mean,
                running_var,
                momentum,
                epsilon,
            ) = args
            if momentum != 0.1:
                raise RuntimeError(f"Expected momenttum=0.1 but got {momentum}")

            shape = meta["val"][0].size()
            dtype = meta["val"][0].dtype
            rank = len(shape)
            running_mean_shape = running_mean.meta["val"].shape
            running_mean_reshaped_shape = [1] * rank
            running_mean_reshaped_shape[1] = running_mean_shape[0]
            epsilon_reshaped_shape = [1] * rank

            sub, add, rsqrt, mul, view, full = get_bn_decomposition(node.target)
            with graph_module.graph.inserting_before(node):
                mean_reshaped = create_node(
                    graph_module.graph,
                    view,
                    args=(running_mean, running_mean_reshaped_shape),
                )
                op1 = create_node(
                    graph_module.graph, sub, args=(activations, mean_reshaped)
                )
                full = create_node(
                    graph_module.graph,
                    full,
                    args=(epsilon_reshaped_shape, epsilon),
                    kwargs={"dtype": dtype},
                )
                var_reshaped = create_node(
                    graph_module.graph,
                    view,
                    args=(running_var, running_mean_reshaped_shape),
                )
                op2 = create_node(graph_module.graph, add, args=(var_reshaped, full))
                op3 = create_node(graph_module.graph, rsqrt, args=(op2,))
                op4 = create_node(graph_module.graph, mul, args=(op1, op3))
                if weights is not None:
                    weights_reshaped = create_node(
                        graph_module.graph,
                        view,
                        args=(weights, running_mean_reshaped_shape),
                    )
                    op5 = create_node(
                        graph_module.graph, mul, args=(op4, weights_reshaped)
                    )
                else:
                    op5 = op4
                output = op5
                if bias is not None:
                    bias_reshaped_shape = running_mean_reshaped_shape
                    bias_reshaped = create_node(
                        graph_module.graph, view, args=(bias, bias_reshaped_shape)
                    )
                    output = create_node(
                        graph_module.graph, add, args=(op5, bias_reshaped)
                    )

                users = [user for user in node.users if node != user]
                node.replace_all_uses_with(output)
                for user in users:
                    if user.target == operator.getitem:
                        user.replace_all_uses_with(output)
                graph_module.graph.erase_node(node)
                graph_module.graph.eliminate_dead_code()
            modified = True
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
