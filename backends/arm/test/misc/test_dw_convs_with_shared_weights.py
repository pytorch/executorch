# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

import torch
from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
)

input_t = Tuple[torch.Tensor]


class DWConvsModule(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        conv = torch.nn.Conv2d(6, 6, kernel_size=(2, 2), groups=6)
        relu = torch.nn.ReLU()
        self.sequential = torch.nn.ModuleList([conv, relu, conv])

    def forward(self, x) -> torch.Tensor:
        for m in self.sequential:
            x = m(x)
        return x

    def get_inputs(self) -> input_t:
        return (torch.randn(1, 6, 24, 24),)


def test_convs_tosa_fp():
    module = DWConvsModule()
    pipeline = TosaPipelineFP[input_t](
        module, module.get_inputs(), aten_op=[], exir_op=[]
    )
    pipeline.run()


def test_convs_tosa_int():
    module = DWConvsModule()
    pipeline = TosaPipelineINT[input_t](
        module, module.get_inputs(), aten_op=[], exir_op=[]
    )
    pipeline.run()


def test_rewrite_conv_pass():
    module = DWConvsModule()
    pipeline = PassPipeline(
        module, module.get_inputs(), passes_with_exported_program=[RewriteConvPass]
    )
    # We can't run TOSA backend dialect operators in eager mode
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
