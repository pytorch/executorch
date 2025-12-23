# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import ClassVar, Dict, Tuple

import torch
from executorch.backends.arm._passes.convert_to_clamp_pass import ConvertToClampPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Input x


class HardTanh(torch.nn.Module):
    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.rand(1, 64, 64, 3),)}

    def __init__(self):
        super().__init__()

        self.hardtanh = torch.nn.Hardtanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh(x)


class ReLU(torch.nn.Module):
    test_data: ClassVar[Dict[str, input_t]] = {"rand": (torch.rand(1, 64, 64, 3),)}

    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


"""
Tests the ConvertToClampPass which converts hardtanh.default and relu.default to clamp.default
"""


@common.parametrize("test_data", HardTanh.test_data)
def test_convert_to_clamp_tosa_FP_hardtahn(test_data: input_t) -> None:
    module = HardTanh()
    op_checks_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
    }
    op_checks_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
    }
    op_checks_not_after_pass = [
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
    ]
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=False,
        ops_before_pass=op_checks_before_pass,
        ops_after_pass=op_checks_after_pass,
        ops_not_after_pass=op_checks_not_after_pass,
        pass_list=[ConvertToClampPass],
    )
    pipeline.run()


@common.parametrize("test_data", ReLU.test_data)
def test_convert_to_clamp_tosa_FP_relu(test_data: input_t) -> None:
    module = ReLU()
    op_checks_before_pass = {
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
    }
    op_checks_after_pass = {
        "executorch_exir_dialects_edge__ops_aten_clamp_default": 1,
    }
    op_checks_not_after_pass = [
        "executorch_exir_dialects_edge__ops_aten_relu_default",
    ]
    pipeline = PassPipeline[input_t](
        module,
        test_data,
        quantize=False,
        ops_before_pass=op_checks_before_pass,
        ops_after_pass=op_checks_after_pass,
        ops_not_after_pass=op_checks_not_after_pass,
        pass_list=[ConvertToClampPass],
    )
    pipeline.run()
