# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from executorch.backends.arm.quantizer import TOSAQuantizer
from executorch.backends.test.harness.stages.quantize import Quantize

from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)

from torch.export import export


class ArmQuantize(Quantize):

    def run(
        self, artifact: torch.nn.Module, inputs: Optional[Tuple[torch.Tensor]]
    ) -> None:
        assert inputs is not None
        if self.is_qat:
            artifact.train()
        captured_graph = export(artifact, inputs, strict=True).module()

        if not isinstance(self.quantizer, TOSAQuantizer):
            raise ValueError("ArmQuantizer can only run with TOSAQuantizer.")

        if self.calibration_samples is not None:
            converted = self.quantizer.quantize_with_submodules(
                captured_graph, self.calibration_samples, bool(self.is_qat)  # type: ignore
            )
        else:
            converted = self.quantizer.quantize_with_submodules(
                captured_graph, [inputs], bool(self.is_qat)
            )

        DuplicateDynamicQuantChainPass()(converted)

        self.converted_graph = converted
