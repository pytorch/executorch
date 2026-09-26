# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.cortex_m.passes.cortex_m_pass import CortexMPass
from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    required_cmsis_nn_buffer_sizes,
)
from executorch.exir.memory import alloc
from executorch.exir.pass_base import PassResult


class InitializeScratchBuffersPass(CortexMPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            sizes = required_cmsis_nn_buffer_sizes(node, self.target_config.backend)
            if sizes is None:
                continue
            for index, size in enumerate(reversed(sizes)):
                scratch = node.args[-(index + 1)]
                if not isinstance(scratch, torch.fx.Node) or scratch.target != alloc:
                    raise RuntimeError(
                        f"Expected scratch alloc node as final argument(s) for {node.target}, got {scratch}."
                    )
                scratch.args = (((size,), torch.uint8),)
                scratch.meta["val"] = torch.empty(
                    (size,), dtype=torch.uint8, device="meta"
                )
                modified = True
        if modified:
            graph_module.recompile()
        return PassResult(graph_module, modified)
