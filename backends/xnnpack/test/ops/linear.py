# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from itertools import product
from typing import Optional

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.backends.xnnpack.test.tester import Quantize, Tester
from executorch.backends.xnnpack.test.tester.tester import Partition

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import QuantizationConfig


class TestLinear(unittest.TestCase):
    def test_fp16_linear(self):
        for use_bias in (True, False):
            self._test_linear(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                uses_bias=use_bias,
                dtype=torch.float16,
                atol=5e-2,
            )

    def test_fp32_linear(self):
        for use_bias in (True, False):
            self._test_linear(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                uses_bias=use_bias,
            )

    def test_fp32_addmm(self):
        """
        Note that the ConvertToLinear pass requires the weight matrix to be transposed.
        """

        class AddMMModule(torch.nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()
                self.mat = torch.nn.Parameter(torch.randn(out_size, in_size))
                self.bias = torch.nn.Parameter(torch.randn(1, out_size))

            def forward(self, x):
                return torch.addmm(self.bias, x, torch.transpose(self.mat, 0, 1))

        self._test_linear(
            lambda in_size, out_size: AddMMModule(in_size, out_size),
            uses_bias=True,
        )

    def test_qs8_linear(self):
        for use_bias in (True, False):
            self._test_linear(
                lambda in_size, out_size: torch.nn.Linear(
                    in_size, out_size, bias=use_bias  # noqa
                ),
                uses_bias=use_bias,
            )

    @unittest.skip("XNNPACK currently only supports per-channel dynamic quantization.")
    def test_qd8_per_tensor_linear(self):
        for uses_bias in (False, True):
            inputs = (torch.randn(2, 4),)
            module = torch.nn.Linear(4, 5, bias=uses_bias)

            self._test_dqlinear(
                module,
                inputs,
                is_per_channel=False,
                uses_bias=uses_bias,
            )

    def test_qd8_per_channel_linear(self):
        for uses_bias in (False, True):
            inputs = (torch.randn(2, 4),)
            module = torch.nn.Linear(4, 5, bias=uses_bias)

            self._test_dqlinear(
                module,
                inputs,
                is_per_channel=True,
                uses_bias=uses_bias,
            )

    @staticmethod
    def _get_4b_dqconfig() -> QuantizationConfig:
        """
        Returns a QuantizationConfig for 4b dynamic quantization for XNNPACK.
        """
        qconfig: QuantizationConfig = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
            weight_qmin=-8,
            weight_qmax=7,
        )
        return qconfig

    def test_qd8_per_channel_4w_linear(self):
        qconfig = self._get_4b_dqconfig()
        input_channels = [2, 63]
        output_channels = [1, 8, 127]
        batches = [1, 2]
        use_bias = [False, True]

        for bs, bias, ipc, opc in product(
            batches,
            use_bias,
            input_channels,
            output_channels,
        ):
            inputs = (torch.rand(bs, ipc),)
            module = torch.nn.Linear(ipc, opc, bias=bias)

            self._test_dqlinear(
                module,
                inputs,
                is_per_channel=True,
                uses_bias=bias,
                qconfig=qconfig,
            )

    def test_qd8_per_channel_linear_parallel(self):
        in_size = 1
        input_size = 4
        output_size = 5

        class ParallelLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear1_bias = torch.nn.Parameter(torch.rand(output_size))

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(output_size, input_size)
                )
                self.linear2_bias = torch.nn.Parameter(torch.rand(output_size))

            def forward(self, x, y):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    y, self.linear2_weight, self.linear2_bias
                )
                return a + b

        inputs = (
            torch.rand(in_size, input_size, dtype=torch.float),
            torch.rand(in_size, input_size, dtype=torch.float),
        )

        self._test_dqlinear(
            ParallelLinear(),
            inputs,
            linear_count=2,
            is_per_channel=True,
            uses_bias=True,
        )

    def test_qd8_per_channel_linear_sequential(self):
        in_size = 1
        input_size = 4
        intermediate_size = 5
        output_size = 3

        class LinearSequential(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear1_bias = torch.nn.Parameter(torch.rand(intermediate_size))

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(output_size, intermediate_size)
                )
                self.linear2_bias = torch.nn.Parameter(torch.rand(output_size))

            def forward(self, x):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    a, self.linear2_weight, self.linear2_bias
                )
                return b

        inputs = (torch.rand(in_size, input_size, dtype=torch.float),)

        self._test_dqlinear(
            LinearSequential(),
            inputs,
            linear_count=2,
            is_per_channel=True,
            uses_bias=True,
        )

    def test_qd8_per_channel_linear_parellel_and_sequential(self):
        in_size = 1
        input_size = 4
        intermediate_size = 5
        output_size = 3

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear1_bias = torch.nn.Parameter(torch.rand(intermediate_size))

                self.linear2_weight = torch.nn.Parameter(
                    torch.rand(intermediate_size, input_size)
                )
                self.linear2_bias = torch.nn.Parameter(torch.rand(intermediate_size))

                self.linear3_weight = torch.nn.Parameter(
                    torch.rand(output_size, intermediate_size)
                )
                self.linear3_bias = torch.nn.Parameter(torch.rand(output_size))

            def forward(self, x, y):
                a = torch.nn.functional.linear(
                    x, self.linear1_weight, self.linear1_bias
                )
                b = torch.nn.functional.linear(
                    y, self.linear2_weight, self.linear2_bias
                )
                c = torch.nn.functional.linear(
                    b, self.linear3_weight, self.linear3_bias
                )
                return (a, c)

        inputs = (
            torch.rand(in_size, input_size, dtype=torch.float),
            torch.rand(in_size, input_size, dtype=torch.float),
        )

        self._test_dqlinear(
            LinearModule(), inputs, linear_count=3, is_per_channel=True, uses_bias=True
        )

    def test_fp32_linear_fused_relu(self):
        class LinearReluModule(torch.nn.Module):
            def __init__(self, in_size, out_size, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_size, out_size, bias=use_bias)

            def forward(self, x):
                return torch.nn.functional.relu(self.linear(x))

        for use_bias in (True, False):
            self._test_linear(
                lambda in_size, out_size: LinearReluModule(
                    in_size,
                    out_size,
                    use_bias,  # noqa
                ),
                uses_bias=use_bias,
            )

    def test_qs8_linear_fused_relu(self):
        class LinearReluModule(torch.nn.Module):
            def __init__(self, in_size, out_size, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_size, out_size, bias=use_bias)

            def forward(self, x):
                return torch.nn.functional.relu(self.linear(x))

        for use_bias in (True, False):
            self._test_linear(
                lambda in_size, out_size: LinearReluModule(
                    in_size,
                    out_size,
                    use_bias,  # noqa
                ),
                uses_bias=use_bias,
                quant=True,
            )

    def _test_linear(
        self,
        make_module,
        uses_bias,
        quant=False,
        dtype: torch.dtype = torch.float,
        atol=1e-03,
    ):
        aten_op, edge_op = (
            (
                "aten.addmm.default",
                "executorch_exir_dialects_edge__ops_aten_addmm_default",
            )
            if uses_bias
            else (
                "aten.mm.default",
                "executorch_exir_dialects_edge__ops_aten_mm_default",
            )
        )

        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]

        """
        Note that torch.nn.Linear maps to aten.mm.default (no bias) or aten.addmm.default (bias),
        which ares then transformed into aten.linear.default by the ConvertToLinear pass.
        """
        for i, _ in enumerate(in_sizes):
            in_size = int(in_sizes[i])
            input_size = int(input_sizes[i])
            output_size = int(output_sizes[i])
            print(f"Testing {in_size} {input_size} {output_size}")

            module = make_module(input_size, output_size).eval().to(dtype)
            inputs = (torch.randn(in_size, input_size).to(dtype),)

            tester = Tester(module, inputs)

            if quant:
                tester.quantize()

            tester.export()
            tester.check_count({aten_op: 1})
            if quant:
                tester.check(["torch.ops.quantized_decomposed"])

            tester.to_edge()
            tester.check_count({edge_op: 1})

            tester.partition()
            tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            tester.check_not([edge_op])

            if quant:
                tester.check_not([edge_op, "torch.ops.quantized_decomposed"])

            tester.to_executorch()
            tester.serialize()
            tester.run_method()
            tester.compare_outputs(qtol=quant, atol=atol)
            print("success")

    def _test_dqlinear(
        self,
        module,
        inputs,
        linear_count=1,
        is_per_channel=False,
        uses_bias=False,
        qconfig: Optional[QuantizationConfig] = None,
    ):
        aten_op, edge_op = (
            (
                "aten.addmm.default",
                "executorch_exir_dialects_edge__ops_aten_addmm_default",
            )
            if uses_bias
            else (
                "aten.mm.default",
                "executorch_exir_dialects_edge__ops_aten_mm_default",
            )
        )

        quant_config = qconfig or get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_dynamic=True,
        )

        tester = Tester(module, inputs)
        tester.quantize(Quantize(quantization_config=quant_config))

        tester.export()
        tester.check_count({aten_op: linear_count})
        tester.check(["torch.ops.quantized_decomposed"])
        tester.dump_artifact()
        tester.to_edge()
        tester.check_count({edge_op: linear_count})

        tester.partition(
            Partition(partitioner=XnnpackDynamicallyQuantizedPartitioner())
        )
        tester.check(["torch.ops.higher_order.executorch_call_delegate"])
        tester.check_not([edge_op])

        tester.to_executorch()
        tester.serialize()
        tester.run_method()
        tester.compare_outputs(atol=5e-02)
