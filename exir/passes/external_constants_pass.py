# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportedProgram


def external_constants_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    Move all constants to external file.
    """
    for module in ep.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        for node in module.graph.nodes:
            if node.op == "placeholder":
                spec = node.meta.get("spec")
                if isinstance(spec, TensorSpec) and spec.const:
                    node.meta["constant_tag"] = "_default_external_constant"
    return ep
