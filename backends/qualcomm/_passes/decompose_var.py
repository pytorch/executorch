# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class DecomposeVar(ExportPass):
    """
    Decompose aten.var.correction and aten.var.dim into supported primitives:
        var(x, dim) = mean((x - mean(x, dim, keepdim=True))^2, dim, keepdim) * N / (N - correction)

    For var.correction:
        correction is an optional Scalar (default=1, i.e. Bessel's correction)
    For var.dim:
        unbiased=True maps to correction=1, unbiased=False maps to correction=0
    """

    def __init__(self):
        super(DecomposeVar, self).__init__()
        self.var_targets = {
            torch.ops.aten.var.correction,
            torch.ops.aten.var.dim,
        }

    def _get_correction(self, node):
        """Extract the correction factor from node args based on op variant."""
        target = node.target
        if target in {torch.ops.aten.var.correction}:
            # var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False)
            # correction is a kwarg, but in the graph it may appear in kwargs
            correction = node.kwargs.get("correction", None)
            if correction is None:
                correction = 1.0
            return float(correction)
        else:
            # var.dim(Tensor self, int[1]? dim=None, bool unbiased=True, bool keepdim=False)
            unbiased = node.args[2] if len(node.args) > 2 else True
            return 1.0 if unbiased else 0.0

    def _get_dim_and_keepdim(self, node):
        """Extract dim and keepdim from node args based on op variant."""
        target = node.target
        if target in {torch.ops.aten.var.correction}:
            # var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False)
            dim = node.args[1] if len(node.args) > 1 else None
            keepdim = node.kwargs.get("keepdim", False)
            return dim, keepdim
        else:
            # var.dim(Tensor self, int[1]? dim=None, bool unbiased=True, bool keepdim=False)
            dim = node.args[1] if len(node.args) > 1 else None
            keepdim = node.args[3] if len(node.args) > 3 else False
            return dim, keepdim

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in self.var_targets:
                x_node = node.args[0]
                meta = node.meta

                correction = self._get_correction(node)
                dim, keepdim = self._get_dim_and_keepdim(node)

                mean_op = torch.ops.aten.mean.dim
                sub_op = torch.ops.aten.sub.Tensor
                mul_op = torch.ops.aten.mul.Tensor
                # Handle dim=None: reduce over all dimensions
                input_shape = node.args[0].meta["val"].shape
                if dim is None:
                    dim = list(range(len(input_shape)))

                with graph.inserting_before(node):
                    x_val = x_node.meta["val"]

                    # Step 1: mean_x = mean(x, dim, keepdim=True)
                    mean_x_node = graph.create_node(
                        "call_function", mean_op, (x_node, dim, True)
                    )
                    mean_x_node.meta = copy_meta(
                        meta,
                        callback=lambda m, _x=x_val, _d=dim: {
                            **m,
                            "val": _x.mean(_d, keepdim=True),
                        },
                    )

                    # Step 2: diff = x - mean_x
                    diff_node = graph.create_node(
                        "call_function", sub_op, (x_node, mean_x_node)
                    )
                    diff_node.meta = copy_meta(
                        meta, callback=lambda m, _x=x_val: {**m, "val": _x}
                    )

                    # Step 3: sq = diff * diff (more efficient than pow(diff, 2))
                    sq_node = graph.create_node(
                        "call_function", mul_op, (diff_node, diff_node)
                    )
                    sq_node.meta = copy_meta(
                        meta, callback=lambda m, _x=x_val: {**m, "val": _x}
                    )

                    # Step 4: var = mean(sq, dim, keepdim)
                    var_node = graph.create_node(
                        "call_function", mean_op, (sq_node, dim, keepdim)
                    )
                    var_node.meta = copy_meta(meta)

                    # Step 5: Apply correction factor if needed
                    if correction != 0.0:
                        # N = product of sizes along reduced dims
                        n = 1
                        for d in dim:
                            n *= input_shape[d]

                        denom = float(n - correction)
                        # Guard against division by zero (e.g. single-element dim with correction=1).
                        # Using inf matches the native PyTorch behavior where 0 * inf → nan.
                        scale = float("inf") if denom == 0 else float(n) / denom
                        result_node = graph.create_node(
                            "call_function", mul_op, (var_node, scale)
                        )
                        result_node.meta = copy_meta(meta)
                    else:
                        result_node = var_node

                for user in node.users.copy():
                    user.replace_input_with(node, result_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
