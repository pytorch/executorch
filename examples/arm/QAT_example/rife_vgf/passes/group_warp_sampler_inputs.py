# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    flow_offset_grid_sampler_operator_name,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch.fx import GraphModule, Node


def _warp_branch(node: Node) -> tuple[Node, Node, Node | None] | None:
    slice_node = None
    if node.target == exir_ops.backend.tosa.SLICE.default:
        slice_node = node
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            return None
        node = input_node
    if node.target != operator.getitem or node.args[1] != 0:
        return None
    custom = node.args[0]
    if (
        not isinstance(custom, Node)
        or custom.target != exir_ops.backend.tosa.CUSTOM.default
        or custom.kwargs.get("operator_name")
        != flow_offset_grid_sampler_operator_name()
    ):
        return None
    return custom, node, slice_node


class GroupWarpSamplerInputsPass(ArmPass):
    """Group RIFE warps before CONCAT without intermediate TOSA dispatches.

    Keep the C4-to-C3 slices as TOSA operations immediately before CONCAT. Only
    reschedule private warp branches; do not change operators or shaders.

    """

    _passes_required_after: set[type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        """Schedule private warp outputs together immediately before CONCAT.

        Args:
            graph_module (GraphModule): Graph after TOSA operator lowering.

        Returns:
            PassResult: Graph with unchanged operators and updated node order.

        """
        graph = graph_module.graph
        modified = False
        for concat in list(graph.nodes):
            if concat.target != exir_ops.backend.tosa.CONCAT.default:
                continue
            branches = [
                branch
                for node in concat.args[0]
                if (branch := _warp_branch(node)) is not None
            ]
            slices = [branch[2] for branch in branches if branch[2] is not None]
            if len(branches) < 2 or not slices:
                continue
            ordered = list(
                dict.fromkeys(
                    [
                        node
                        for custom, getitem, _ in branches
                        for node in (custom, getitem)
                    ]
                    + slices
                )
            )
            moving = set(ordered)
            if any(
                user not in moving and user is not concat
                for node in moving
                for user in node.users
            ):
                continue
            # Preserve dependency order, including dependencies between warp inputs.
            positions = {node: index for index, node in enumerate(ordered)}
            if any(
                positions[input_node] >= positions[node]
                for node in moving
                for input_node in node.all_input_nodes
                if input_node in moving
            ):
                continue
            current = list(graph.nodes)
            if (
                current[current.index(concat) - len(ordered) : current.index(concat)]
                == ordered
            ):
                continue
            for node in ordered:
                concat.prepend(node)
            modified = True

        if modified:
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
