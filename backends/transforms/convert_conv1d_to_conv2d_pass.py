# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.transforms.utils import (
    get_param_tensor,
    is_param_node,
    set_param_tensor,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertConv1dToConv2dPass(ExportPass):
    """Express Conv1d as Conv2d with a unit-height spatial dimension.

    Only convolutions with lifted parameter, buffer, or constant weights are
    converted. Dynamic weights and unfolded QDQ weights are left unchanged.
    The pass emits squeeze and unsqueeze boundaries for each converted
    convolution; downstream view cleanup can fold adjacent boundaries.
    """

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    def _unsqueeze_weight(self, weight_node: torch.fx.Node) -> bool:
        weight = get_param_tensor(self.exported_program, weight_node)
        if weight is None:
            return False
        if weight.dim() == 4:
            return True
        if weight.dim() != 3:
            return False

        weight_2d = weight.unsqueeze(2).contiguous()
        set_param_tensor(self.exported_program, weight_node, weight_2d)
        weight_node.meta["val"] = weight_node.meta["val"].unsqueeze(2)
        return True

    def _conv1d_weight_node(self, node: torch.fx.Node) -> torch.fx.Node | None:
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.aten.convolution.default
            or len(node.args) != 9
        ):
            return None
        weight_node = node.args[1]
        if not isinstance(weight_node, torch.fx.Node) or not is_param_node(
            self.exported_program, weight_node
        ):
            return None
        weight = get_param_tensor(self.exported_program, weight_node)
        weight_meta = weight_node.meta.get("val")
        if (
            not isinstance(weight, torch.Tensor)
            or weight.dim() != 3
            or not isinstance(weight_meta, torch.Tensor)
            or weight_meta.dim() != 3
        ):
            return None
        return weight_node

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        candidates = {
            node: weight_node
            for node in graph.nodes
            if (weight_node := self._conv1d_weight_node(node)) is not None
        }

        for node, weight_node in candidates.items():
            if any(
                candidates.get(user) is not weight_node
                or any(
                    arg is weight_node
                    for index, arg in enumerate(user.args)
                    if index != 1
                )
                or any(arg is weight_node for arg in user.kwargs.values())
                for user in weight_node.users
            ):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            if not self._unsqueeze_weight(weight_node):
                continue
            with graph.inserting_before(node):
                input_2d = graph.call_function(
                    exir_ops.edge.aten.unsqueeze_copy.default,
                    args=(input_node, 2),
                )

            args = list(node.args)
            args[0] = input_2d
            args[3] = [1, *list(args[3])]  # stride
            args[4] = [0, *list(args[4])]  # padding
            args[5] = [1, *list(args[5])]  # dilation
            args[7] = [0, *list(args[7])]  # output padding
            node.args = tuple(args)

            with graph.inserting_after(node):
                output_1d = graph.call_function(
                    exir_ops.edge.aten.squeeze_copy.dim,
                    args=(node, 2),
                )
            for user in list(node.users):
                if user is not output_1d:
                    user.replace_input_with(node, output_1d)

            modified = True

        if not modified:
            return PassResult(graph_module, False)

        graph_module.recompile()
        result = super().call(graph_module)
        return PassResult(result.graph_module, True)
