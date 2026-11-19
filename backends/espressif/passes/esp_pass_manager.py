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
from executorch.exir.pass_base import ExportedProgramPassResult, ExportPass
from executorch.exir.pass_manager import ExportedProgramPassManager
from executorch.exir.passes.replace_aten_with_edge_pass import OpReplacePass
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.program._program import lift_constant_tensor_pass

from .lower_to_esp import LowerToEsp


class EspPassManager(ExportedProgramPassManager):
    """Lower an isolated copy of a Fused Quant Edge program to ESP-NN or portable ops.

    ``modified`` is conservative: metadata refresh and ATen-to-Edge retracing
    report changes even when the executable graph is unchanged.
    """

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
        result = super().__call__(copy.deepcopy(exported_program), override_verifiers)
        # Scalar normalization can create tensor attributes; direct callers also
        # need these lifted into the program signature before serialization.
        num_inputs = len(result.exported_program.graph_signature.input_specs)
        lift_constant_tensor_pass(result.exported_program)
        self.check(result.exported_program)
        return ExportedProgramPassResult(
            result.exported_program,
            result.modified
            or len(result.exported_program.graph_signature.input_specs) != num_inputs,
        )
