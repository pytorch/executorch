# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import unittest

import torch

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig


class TopK(torch.nn.Module):
    def __init__(self, k=1, dim=-1, output="both") -> None:
        super().__init__()
        self.k = k
        self.dim = dim
        self.output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.topk(x, self.k, dim=self.dim)
        if self.output == "indices":
            return result.indices
        if self.output == "values":
            return result.values
        return tuple(result)


class QuantizedTopK(TopK):
    def __init__(self, k=1, dim=-1, output="both", quantize_values=False):
        super().__init__(k, dim, output)
        self.quantize_values = quantize_values

    @staticmethod
    def _qdq(x):
        quantized = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, 0.1, 0, -128, 127, torch.int8
        )
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            quantized, 0.1, 0, -128, 127, torch.int8
        )

    def forward(self, x):
        result = super().forward(self._qdq(torch.relu(self._qdq(x))))
        if not self.quantize_values or self.output == "indices":
            return result
        if self.output == "values":
            return self._qdq(result)
        return self._qdq(result[0]), result[1]


class TestTopK(unittest.TestCase):
    def test_a8w8_topk_indices_only(self):
        inputs = (torch.randperm(127).float().view(1, 127) * 0.1,)
        tester = SamsungTester(
            QuantizedTopK(k=30, output="indices"),
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_topk_default": 1})
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_a8w8_topk_quantized_values(self):
        inputs = (torch.randperm(127).float().view(1, 127) * 0.1,)
        for output in ("values", "both"):
            with self.subTest(output=output):
                tester = SamsungTester(
                    QuantizedTopK(k=30, output=output, quantize_values=True),
                    inputs,
                    [gen_samsung_backend_compile_spec(TestConfig.chipset)],
                )
                (
                    tester.export()
                    .to_edge_transform_and_lower()
                    .check_count(
                        {"executorch_exir_dialects_edge__ops_aten_topk_default": 1}
                    )
                    .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                    .to_executorch()
                    .run_method_and_compare_outputs(inputs=inputs)
                )

    def _test(self, module: torch.nn.Module, inputs):
        tester = SamsungTester(
            module,
            inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )
        (
            tester.export()
            .check_count({"torch.ops.aten.topk.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"executorch_exir_dialects_edge__ops_aten_topk_default": 1})
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_topk_dim3(self):
        # Random permutation of 1..1024 gives well-separated values,
        # so TopK k-boundary ordering is unambiguous.
        x = torch.randperm(16 * 8 * 8, dtype=torch.float32).view(1, 16, 8, 8)
        inputs = (x,)
        self._test(TopK(k=5, dim=3), inputs)

    def test_fp32_topk_dim_negative1(self):
        x = torch.randperm(16 * 8 * 8, dtype=torch.float32).view(1, 16, 8, 8)
        inputs = (x,)
        self._test(TopK(k=5, dim=-1), inputs)

    def test_fp32_topk_indices_only(self):
        inputs = (-torch.randperm(2100, dtype=torch.float32).view(1, 2100),)
        self._test(TopK(k=300, output="indices"), inputs)

    def test_fp32_topk_values_only(self):
        inputs = (-torch.randperm(2100, dtype=torch.float32).view(1, 2100),)
        self._test(TopK(k=300, output="values"), inputs)
