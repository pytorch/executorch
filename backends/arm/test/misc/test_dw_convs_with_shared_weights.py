# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

import torch
from executorch.backends.arm.test.tester.test_pipeline import (
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


def test_convs_tosa_FP():
    module = DWConvsModule()
    pipeline = TosaPipelineFP[input_t](
        module, module.get_inputs(), aten_op=[], exir_op=[]
    )
    pipeline.run()


def test_convs_tosa_INT():
    module = DWConvsModule()
    pipeline = TosaPipelineINT[input_t](
        module, module.get_inputs(), aten_op=[], exir_op=[]
    )
    pipeline.run()
