# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.exir.pass_base import ExportPass, PassResult


logger = logging.getLogger(__name__)
INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max


class ConvertInt64ConstOpsToInt32Pass(ArmPass):
    """Rewrite constant ops that produce int64 to int32 where safe.

    List of supported operatos:
      1. `torch.full`
      2. `torch.arange`
      3. `torch.eye`
      4. `torch.linspace`
      5. `torch.tensor`

    """

    _passes_required_after: Set[Type[ExportPass]] = {ComputeConstantOpsAOTPass}

    torch_ops = [
        torch.ops.aten.full.default,
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
        torch.ops.aten.eye.default,
        torch.ops.aten.linspace.default,
    ]

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        converted = 0
        skipped = 0
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            if (
                node.target
                not in ComputeConstantOpsAOTPass.targeted_ops + self.torch_ops
            ):
                continue

            data = node.target(*node.args, **node.kwargs)
            if data.dtype is not torch.int64:
                continue

            min_val, max_val = torch.min(data), torch.max(data)
            if INT32_MIN <= min_val and max_val <= INT32_MAX:
                node.update_kwarg("dtype", torch.int32)
                modified = True
                converted += 1
            else:
                skipped += 1

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        if converted or skipped:
            logger.warning(
                "ConvertInt64ConstOpsToInt32Pass: cast %d constant operator(s) "
                "to int32 and skipped %d outside the int32 range.",
                converted,
                skipped,
            )

        return PassResult(graph_module, modified)
