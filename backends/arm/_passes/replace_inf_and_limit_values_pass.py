# Copyright (c) Qualcomm Innovation Center, Inc.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This pass is based on backends/qualcomm/_passes/replace_inf_values.py
# with some modification to replaced inf values.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult


class ReplaceInfAndLimitValuesPass(ArmPass):
    """Rewrites +inf/-inf and floating-point limit values (e.g.,
    torch.finfo(...).min/max) to configured quantization-friendly values,
    improving quantizer stability (notably for attention mask paths).
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(
        self,
        neg_inf: float,
        pos_inf: float,
        tfa_pass: bool = False,
    ):
        super().__init__(tfa_pass=tfa_pass)
        self.neg_inf = neg_inf
        self.pos_inf = pos_inf

    def _allowed_to_transform_named_buffer(self, buf_name, graph_module) -> bool:
        attr_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op == "get_attr" and node.target == buf_name
        ]

        can_transform_buffer = True
        for attr_node in attr_nodes:
            for user in list(attr_node.users):
                if user.op != "call_function":
                    continue
                if not self.allowed_to_transform(NodeMetadata(user.meta)):
                    can_transform_buffer = False
                    break
            if not can_transform_buffer:
                break

        return can_transform_buffer

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for buf_name, tensor in graph_module.named_buffers():
            if not tensor.is_floating_point():
                continue
            if not self._allowed_to_transform_named_buffer(buf_name, graph_module):
                continue

            modified = True

            t = torch.nan_to_num(tensor, posinf=self.pos_inf, neginf=self.neg_inf)
            setattr(graph_module, buf_name, t)

        for node in graph_module.graph.nodes:
            arg_list = list(node.args)
            for index, arg in enumerate(arg_list):
                if arg == float("-inf") or arg == torch.finfo(torch.float32).min:
                    modified = True
                    arg_list[index] = self.neg_inf
                elif arg == float("inf") or arg == torch.finfo(torch.float32).max:
                    modified = True
                    arg_list[index] = self.pos_inf
            node.args = tuple(arg_list)

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
