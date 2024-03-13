# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Dict

import torch
from executorch.exir.pass_base import PassBase, PassResult

replacements: Dict[torch._ops.OpOverloadPacket, torch._ops.OpOverload] = {
    torch.ops.aten.sym_size: torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride: torch.ops.aten.sym_stride.int,
    torch.ops.aten.sym_numel: torch.ops.aten.sym_numel.default,
}


class ReplaceSymSizeOpPass(PassBase):
    """
    Replace torch.ops.aten.sym_size with torch.ops.aten.sym_size.int
    and torch.ops.aten.sym_stride with torch.ops.aten.sym_stride.int

    TODO: this can be refactors into a general OpReplacementPass
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.target in replacements:
                    node.target = replacements[node.target]
        return PassResult(graph_module, True)
