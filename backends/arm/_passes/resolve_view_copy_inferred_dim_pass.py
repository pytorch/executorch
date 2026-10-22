# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.symbolic_shape_utils import materialize_symints
from executorch.backends.arm._passes.symbolic_to_tosa_shape_pass import (
    SymbolicToTosaShapesPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class ResolveViewCopyInferredDimPass(ArmPass):
    """Materialize inferred view dimensions before TOSA shape lowering."""

    _passes_required_after = {SymbolicToTosaShapesPass}
    target_ops = {
        torch.ops.aten.view.default,
        exir_ops.edge.aten.view_copy.default,
    }

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function" or node.target not in self.target_ops:
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], (list, tuple)):
                continue

            shape = node.args[1]
            inferred_dim_indices = [
                i for i, dim in enumerate(shape) if isinstance(dim, int) and dim == -1
            ]
            if not inferred_dim_indices:
                continue
            if len(inferred_dim_indices) > 1:
                raise ValueError("View shape contains more than one inferred dimension")
            if "val" not in node.meta:
                raise ValueError(
                    "Cannot resolve inferred view dimension without metadata"
                )

            output = node.meta["val"]
            output_shape = output.shape if hasattr(output, "shape") else output
            if len(shape) != len(output_shape):
                raise ValueError(
                    "Cannot resolve inferred view dimension when view shape rank "
                    f"does not match output metadata rank: {len(shape)} != "
                    f"{len(output_shape)}"
                )

            inferred_dim_index = inferred_dim_indices[0]
            with graph.inserting_before(node):
                (inferred_dim,) = materialize_symints(
                    graph, [output_shape[inferred_dim_index]]
                )

            resolved_shape = list(shape)
            resolved_shape[inferred_dim_index] = inferred_dim
            new_args = list(node.args)
            new_args[1] = (
                tuple(resolved_shape) if isinstance(shape, tuple) else resolved_shape
            )
            node.args = tuple(new_args)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
        return PassResult(graph_module, modified)
