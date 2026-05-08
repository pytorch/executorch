# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Test the pad_constant_nd op which pads the input tensor at specific dimension(s).
#
from typing import Tuple

import torch
import torch.nn.functional as F
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.pad.default"
exir_op = "executorch_exir_dialects_edge__ops_aten_pad_default"

input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    "4dim_last1dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        1,
        "constant",
    ),
    "4dim_last2dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 0, 1, 0, 0, 0, 0, 0),
        2,
        "constant",
    ),
    "4dim_last3dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 1, 0, 2, 0, 2, 0, 0),
        3,
        "constant",
    ),
    "4dim_last4dim": lambda: (
        torch.rand(1, 1, 16, 16),
        (1, 0, 1, 1, 0, 2, 0, 2),
        4,
        "constant",
    ),
    "3dim_last1dim": lambda: (torch.rand(1, 1, 16), (1, 1, 0, 0, 0, 0), 1, "constant"),
    "3dim_last2dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 1, 0, 0), 2, "constant"),
    "3dim_last3dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 0, 1, 1), 3, "constant"),
    "2dim_last1dim": lambda: (torch.rand(1, 1, 16), (1, 1, 0, 0), 1, "constant"),
    "2dim_last2dim": lambda: (torch.rand(1, 1, 16), (1, 0, 1, 1), 2, "constant"),
    "4dim_reflect": lambda: (
        torch.rand(6, 6, 6, 6),
        (3, 3, 3, 3, 3, 3),
        None,
        "reflect",
    ),
    "4dim_replicate": lambda: (
        torch.rand(3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3),
        None,
        "replicate",
    ),
    "4dim_circular": lambda: (
        torch.rand(3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3),
        None,
        "circular",
    ),
    "2dim_reflect": lambda: (
        torch.rand(6, 6),
        (3, 3),
        None,
        "reflect",
    ),
    "2dim_replicate": lambda: (
        torch.rand(3, 3),
        (3, 3),
        None,
        "replicate",
    ),
    "2dim_circular": lambda: (
        torch.rand(3, 3),
        (3, 3),
        None,
        "circular",
    ),
}

test_data_suite_bf16 = {
    "4dim_last1dim_bf16": lambda: (
        torch.rand(1, 1, 8, 8, dtype=torch.bfloat16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        1.0,
        "constant",
    ),
    "3dim_last1dim_bf16": lambda: (
        torch.rand(1, 1, 8, dtype=torch.bfloat16),
        (1, 0, 1, 0, 0, 0),
        -0.5,
        "constant",
    ),
}
test_data_suite_fp16 = {
    "4dim_last1dim_fp16": lambda: (
        torch.rand(1, 1, 8, 8, dtype=torch.float16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        1.0,
        "constant",
    ),
    "3dim_last1dim_fp16": lambda: (
        torch.rand(1, 1, 8, dtype=torch.float16),
        (1, 0, 1, 0, 0, 0),
        -0.5,
        "constant",
    ),
}


class ConstantPadND(torch.nn.Module):
    def __init__(
        self,
        pad: Tuple,
        value: float | None = None,
        mode: str = "constant",
    ):
        super().__init__()
        self.value = value
        self.mode = mode
        nonzero_idx = len(pad)
        for i in range(0, len(pad), 2):
            if pad[i] + pad[i + 1] == 0:
                nonzero_idx = i
                break
        self.pad = pad[:nonzero_idx]

    def forward(self, x: torch.Tensor):
        return F.pad(x, pad=self.pad, mode=self.mode, value=self.value)


@common.parametrize(
    "test_data",
    test_data_suite | test_data_suite_bf16 | test_data_suite_fp16,
)
def test_constant_pad_nd_tosa_FP(test_data: Tuple):
    test_data, padding, value, mode = test_data()
    pipeline = TosaPipelineFP[input_t1](
        ConstantPadND(padding, value, mode),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_constant_pad_nd_tosa_INT(test_data: Tuple):
    test_data, padding, value, mode = test_data()
    pipeline = TosaPipelineINT[input_t1](
        ConstantPadND(padding, value, mode),
        (test_data,),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_constant_pad_nd_tosa_INT_a16w8(test_data: Tuple):
    """Test constant_pad_nd op with int16 I/O quantization for TOSA INT."""
    test_data, padding, value, mode = test_data()
    pipeline = TosaPipelineINT[input_t1](
        ConstantPadND(padding, value, mode),
        (test_data,),
        aten_op,
        exir_op,
        tosa_extensions=["int16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite | test_data_suite_fp16)
@common.SkipIfNoModelConverter
def test_constant_pad_nd_vgf_no_quant(test_data: Tuple):
    inp, padding, value, mode = test_data()
    pipeline = VgfPipeline[input_t1](
        ConstantPadND(padding, value, mode),
        (inp,),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_constant_pad_nd_vgf_quant(test_data: Tuple):
    inp, padding, value, mode = test_data()
    pipeline = VgfPipeline[input_t1](
        ConstantPadND(padding, value, mode),
        (inp,),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_constant_pad_nd_vgf_quant_a16w8(test_data: Tuple):
    inp, padding, value, mode = test_data()
    pipeline = VgfPipeline[input_t1](
        ConstantPadND(padding, value, mode),
        (inp,),
        aten_op,
        exir_op,
        quantize=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(get_symmetric_a16w8_quantization_config())
    pipeline.run()
