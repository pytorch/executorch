# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List

import torch
import torch.fx
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.passes import dead_code_elimination_pass


class RemoveCloneOpsTransformImported(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        finalize_passes: List[PassType] = [
            RemoveCloneOpsTransform(eliminate_quant_dequant_pairs=False),
        ]
        result = PassManager(passes=finalize_passes)(graph_module)
        dead_code_elimination_pass(result.graph_module)
        return result
