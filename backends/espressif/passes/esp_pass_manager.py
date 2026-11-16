# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.transforms.fused_quant.decompose_fused_quant import (
    AssertNoFusedQuantOps,
    DecomposeFusedQuant,
)
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.exir.pass_manager import ExportedProgramPassManager

from .lower_to_esp import LowerToEsp


class EspPassManager(ExportedProgramPassManager):
    """Lower fused-quant Edge operators to ESP-NN or portable kernels."""

    def __init__(self) -> None:
        super().__init__(
            passes=[
                LowerToEsp(),
                DecomposeFusedQuant(),
                RemoveUnusedConstantsPass(),
                AssertNoFusedQuantOps(),
            ]
        )
