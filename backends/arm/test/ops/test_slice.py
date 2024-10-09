# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.xnnpack.test.tester.tester import Quantize
from parameterized import parameterized


class TestSimpleSlice(unittest.TestCase):

    class Slice(torch.nn.Module):

        sizes = [(10), (10, 10), (10, 10, 10), ((1, 12, 10, 10))]
        test_tensors = [(torch.ones(n),) for n in sizes]

        def forward(self, x: torch.Tensor):
            if x.dim() == 1:
                return x[3:-3]
            elif x.dim() == 2:
                return x[1:3, 3:5]
            elif x.dim() == 3:
                return x[0:7, 0:1, 0:8]
            elif x.dim() == 4:
                return x[:, 2:5, 3:5, 4:10]

    def _test_slice_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .check(["torch.ops.aten.slice.Tensor"])
            .to_edge()
            .check(["executorch_exir_dialects_edge__ops_aten_slice_copy"])
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    def _test_slice_tosa_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor], permute: bool
    ):

        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec(
                    permute_memory_to_nhwc=permute
                ),
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check(["torch.ops.aten.slice.Tensor"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data, qtol=1)
        )

    def _test_slice_u55_BI_pipeline(
        self, module: torch.nn.Module, test_data: Tuple[torch.Tensor]
    ):
        quantizer = ArmQuantizer().set_io(get_symmetric_quantization_config())
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_u55_compile_spec(),
            )
            .quantize(Quantize(quantizer, get_symmetric_quantization_config()))
            .export()
            .check(["torch.ops.aten.slice.Tensor"])
            .to_edge()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
        )

    @parameterized.expand(Slice.test_tensors)
    def test_slice_tosa_MI(self, tensor):
        self._test_slice_tosa_MI_pipeline(self.Slice(), (tensor,))

    @parameterized.expand(Slice.test_tensors[:2])
    def test_slice_nchw_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_slice_tosa_BI_pipeline(self.Slice(), (test_tensor,), False)

    @parameterized.expand(Slice.test_tensors[2:])
    def test_slice_nhwc_tosa_BI(self, test_tensor: torch.Tensor):
        self._test_slice_tosa_BI_pipeline(self.Slice(), (test_tensor,), True)

    # Fails during Vela compilation when trying to use a Tuple as a Named tuple,
    # Could be Vela Issue, wait until Regor.
    @parameterized.expand(Slice.test_tensors)
    def test_slice_u55_BI(self, test_tensor: torch.Tensor):
        self._test_slice_u55_BI_pipeline(self.Slice(), (test_tensor,))
