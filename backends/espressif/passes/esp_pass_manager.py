# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from executorch.backends.transforms.fused_quant.decompose_fused_quant import (
    AssertNoFusedQuantOps,
    DecomposeFusedQuant,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.transforms.fused_quant.passes import get_fused_quant_passes
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import ExportedProgramPassManager
from executorch.exir.passes.replace_aten_with_edge_pass import OpReplacePass
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.program._program import lift_constant_tensor_pass

from torch.fx import GraphModule

from .lower_to_esp import LowerToEsp


class EspPassManager(ExportedProgramPassManager):
    """Lower fused-quant Edge operators to ESP-NN or portable kernels."""

    def __init__(self) -> None:
        super().__init__(
            passes=[
                *get_fused_quant_passes(channels_last=True),
                # Refresh contiguous tensor metadata after the layout rewrites.
                ExportPass(),
                LowerToEsp(),
                ToChannelsFirst(),
                DecomposeFusedQuant(),
                ScalarToTensorPass(),
                OpReplacePass(),
                RemoveUnusedConstantsPass(),
                AssertNoFusedQuantOps(),
            ]
        )

    def __call__(self, exported_program, override_verifiers=None):
        program = copy.copy(exported_program)
        # The pass manager copies programs shallowly; decomposition also retires
        # qparam constants, so isolate both graph and constant dictionaries.
        graph = copy.deepcopy(program.graph)
        for original, cloned in zip(program.graph.nodes, graph.nodes):
            cloned.name = original.name
        module = GraphModule(program.graph_module, graph)
        module.meta = program.graph_module.meta.copy()
        program._graph_module = module
        program._graph_signature = copy.deepcopy(program.graph_signature)
        program._state_dict = program.state_dict.copy()
        program._constants = program.constants.copy()

        result = super().__call__(program, override_verifiers)
        lift_constant_tensor_pass(result.exported_program)
        return result
