# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch

from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_conv_pass import (
    FuseBatchNormWithConvPass,
)
from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_linear_pass import (
    FuseBatchNormWithLinearPass,
)
from executorch.backends.nxp.aten_passes.fuse_linear_and_add_pass import (
    FuseLinearAndAddPass,
)
from executorch.backends.nxp.aten_passes.remove_nodes_with_known_outputs import (
    RemoveNodesWithKnownOutputs,
)
from executorch.backends.nxp.aten_passes.split_group_convolution import (
    SplitGroupConvolution,
)
from executorch.backends.nxp.aten_passes.split_gru_based_on_num_layers import (
    SplitGRUBasedOnNumLayers,
)
from executorch.exir.pass_manager import PassManager
from torch import nn
from torch.fx.passes.infra.pass_base import PassResult

PassType = type[Callable[[torch.fx.GraphModule], PassResult]]


class NeutronAtenPassManager(PassManager):

    def __init__(self, passes: list[PassType] = None):
        passes: list[PassType] = passes or [
            FuseBatchNormWithConvPass(),
            FuseBatchNormWithLinearPass(),
            SplitGroupConvolution(),
            SplitGRUBasedOnNumLayers(),
            RemoveNodesWithKnownOutputs(),
            FuseLinearAndAddPass(),
        ]

        super().__init__(passes)

    def __call__(self, module: nn.Module) -> PassResult:
        pass_result: PassResult = super().__call__(module)

        graph_module = pass_result.graph_module
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return pass_result
