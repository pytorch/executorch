# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)

from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeBatchNormNoStatsPass(ArmPass):
    """
    Decompose BatchNorm2d(track_running_stats=False) (aten._native_batch_norm_legit_no_training)
    into a sequence of elementwise operations:

    # let input = x, rm = running_mean, rv = running_var, eps: float
    rm_view    = view(rm, weights_shape)
    rv_view    = view(rv, weights_shape)
    centered   = sub(x, rm_view)
    eps_full   = full(eps_shape, eps)
    var_eps    = add(rv_view, eps_full)
    inv_sqrt   = rsqrt(var_eps)
    normed     = mul(centered, inv_sqrt)
    weighted   = mul(normed, w_view)
    biased     = add(weighted, b_view)

    Source: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """

    _passes_required_after: Set[Type[ExportPass]] = {
        ComputeConstantOpsAOTPass,
        InsertTableOpsPass,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        bn_ops = (
            exir_ops.edge.aten._native_batch_norm_legit.no_stats,
            exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
            torch.ops.aten._native_batch_norm_legit_no_training.default,
            torch.ops.aten.batch_norm.default,
            torch.ops.aten.native_batch_norm.default,
        )

        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target not in bn_ops
                or not self.allowed_to_transform(node.meta)
            ):
                continue

            if node.target in (
                torch.ops.aten.batch_norm.default,
                torch.ops.aten.native_batch_norm.default,
            ):
                # signature: (input, weight, bias, mean, var, training, momentum, eps, cudnn_enabled)
                # pos‐arg 5 is training
                training = node.kwargs.get("training", False)
                if len(node.args) > 5:
                    training = node.args[5]
                if training:
                    # skip training‐mode batchnorm
                    continue

            # Extract args
            args = node.args
            meta = node.meta

            # Default eps
            eps: float = torch.finfo().eps
            # weight and bias may be None
            x = args[0]
            weight = args[1] if len(args) > 1 else None
            bias = args[2] if len(args) > 2 else None
            running_mean = args[3]
            running_var = args[4]
            if len(args) > 6:
                eps = args[6]

            # Determine shapes
            val = meta.get("val")
            ref_tensor = val[0] if isinstance(val, tuple) else val
            shape = tuple(ref_tensor.size())
            dtype = ref_tensor.dtype
            rank = len(shape)

            # channel dimension is 1 for BatchNorm2d
            channel_axis = 1
            weights_shape = [1] * rank
            weights_shape[channel_axis] = shape[channel_axis]
            num_features = shape[channel_axis]

            # Ops to use
            sub_op = exir_ops.edge.aten.sub.Tensor
            view_op = exir_ops.edge.aten.view_copy.default
            full_op = exir_ops.edge.aten.full.default
            add_op = exir_ops.edge.aten.add.Tensor
            rsqrt_op = exir_ops.edge.aten.rsqrt.default
            mul_op = exir_ops.edge.aten.mul.Tensor

            # Begin decomposition
            with graph_module.graph.inserting_before(node):
                # reshape running stats
                rm_view = create_node(
                    graph_module.graph,
                    view_op,
                    args=(running_mean, weights_shape),
                    from_node=node,
                )
                rv_view = create_node(
                    graph_module.graph,
                    view_op,
                    args=(running_var, weights_shape),
                    from_node=node,
                )
                # center input
                centered = create_node(
                    graph_module.graph,
                    sub_op,
                    args=(x, rm_view),
                    from_node=node,
                )
                # epsilon tensor
                eps_shape = [1] * rank
                eps_full = create_node(
                    graph_module.graph,
                    full_op,
                    args=(eps_shape, eps),
                    kwargs={"dtype": dtype},
                    from_node=node,
                )
                # var + eps
                var_eps = create_node(
                    graph_module.graph,
                    add_op,
                    args=(rv_view, eps_full),
                    from_node=node,
                )
                # inverse sqrt
                inv_sqrt = create_node(
                    graph_module.graph,
                    rsqrt_op,
                    args=(var_eps,),
                    from_node=node,
                )
                # normalized
                normed = create_node(
                    graph_module.graph,
                    mul_op,
                    args=(centered, inv_sqrt),
                    from_node=node,
                )

                # weight
                if weight is None:
                    one = create_node(
                        graph_module.graph,
                        full_op,
                        args=([num_features], 1),
                        kwargs={"dtype": dtype},
                        from_node=node,
                    )
                    w_view = create_node(
                        graph_module.graph,
                        view_op,
                        args=(one, weights_shape),
                        from_node=node,
                    )
                else:
                    w_view = create_node(
                        graph_module.graph,
                        view_op,
                        args=(weight, weights_shape),
                        from_node=node,
                    )
                weighted = create_node(
                    graph_module.graph,
                    mul_op,
                    args=(normed, w_view),
                    from_node=node,
                )

                # bias
                if bias is None:
                    zero = create_node(
                        graph_module.graph,
                        full_op,
                        args=([num_features], 0),
                        kwargs={"dtype": dtype},
                        from_node=node,
                    )
                    b_view = create_node(
                        graph_module.graph,
                        view_op,
                        args=(zero, weights_shape),
                        from_node=node,
                    )
                else:
                    b_view = create_node(
                        graph_module.graph,
                        view_op,
                        args=(bias, weights_shape),
                        from_node=node,
                    )
                final_out = create_node(
                    graph_module.graph,
                    add_op,
                    args=(weighted, b_view),
                    from_node=node,
                )

                users = [u for u in node.users if u is not node]
                node.replace_all_uses_with(final_out)
                for u in users:
                    if u.target == operator.getitem:
                        u.replace_all_uses_with(final_out)
                graph_module.graph.erase_node(node)
                graph_module.graph.eliminate_dead_code()

        graph_module.recompile()
        new_gm = super().call(graph_module).graph_module
        return PassResult(new_gm, True)
