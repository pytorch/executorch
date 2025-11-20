# Copyright (c) Qualcomm Innovation Center, Inc.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This pass is based on backends/qualcomm/_passes/replace_inf_values.py
# with some modification to replaced inf values.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.pass_base import ExportPass, PassResult


class ReplaceInfValuesPass(ArmPass):
    """
    Due to limitation in Quantizer, we need to change inf/-inf to more quantizable values.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for buf_name, tensor in graph_module.named_buffers():
            if tensor.is_floating_point():
                modified = True
                # 255 here is mainly for attention_mask in Llama for reasonable quant scale
                tensor[tensor == float("inf")] = 255
                tensor[tensor == float("-inf")] = -255
                setattr(graph_module, buf_name, tensor)

        for node in graph_module.graph.nodes:
            arg_list = list(node.args)
            for index, arg in enumerate(arg_list):
                if arg == float("-inf"):
                    modified = True
                    arg_list[index] = -255
                elif arg == float("inf"):
                    modified = True
                    arg_list[index] = +255
            node.args = tuple(arg_list)

        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
