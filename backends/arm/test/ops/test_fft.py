# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    VgfPipeline,
)

aten_op = "torch.ops.aten.fft_rfft2.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_fft_rfft2_default"

input_t1 = Tuple[torch.Tensor]


class RFFT2D(torch.nn.Module):
    test_parameters = {
        "rank2": lambda: (torch.randn(8, 16),),
        "rank3": lambda: (torch.randn(2, 8, 16),),
        "rank4": lambda: (torch.randn(1, 2, 8, 16),),
        "ones": lambda: (torch.ones(2, 8, 16),),
        "zeros": lambda: (torch.zeros(2, 8, 16),),
    }

    def forward(self, x: torch.Tensor):
        output = torch.fft.rfft2(x)
        return output.real, output.imag


@common.parametrize("test_data", RFFT2D.test_parameters)
def test_rfft2d_tosa_FP(test_data: input_t1):
    pipeline = TosaPipelineFP[input_t1](
        RFFT2D(),
        test_data(),
        aten_op,
        exir_op,
        run_on_tosa_ref_model=False,
        tosa_version="1.1",
        tosa_extensions=["fft"],
    )
    pipeline.run()


@common.parametrize("test_data", RFFT2D.test_parameters)
@common.SkipIfNoModelConverter
def test_rfft2d_vgf_no_quant(test_data: input_t1):
    pipeline = VgfPipeline[input_t1](
        RFFT2D(),
        test_data(),
        aten_op,
        exir_op,
        run_on_vulkan_runtime=False,
        quantize=False,
        tosa_version="TOSA-1.1+FP",
        tosa_extensions=["fft"],
    )
    pipeline.run()
