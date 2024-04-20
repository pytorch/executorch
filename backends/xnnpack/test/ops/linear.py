# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from itertools import product
from typing import Optional, Tuple

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
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    num_batch_dims=num_batch_dims,
                    uses_bias=use_bias,
                    dtype=torch.float16,
                    atol=5e-2,
                )

    def test_fp32_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    uses_bias=use_bias,
                    num_batch_dims=num_batch_dims,
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

    def test_fp32_linear_fused_relu(self):
        class LinearReluModule(torch.nn.Module):
            def __init__(self, in_size, out_size, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_size, out_size, bias=use_bias)

            def forward(self, x):
                return torch.nn.functional.relu(self.linear(x))

        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: LinearReluModule(
                        in_size,
                        out_size,
                        use_bias,  # noqa
                    ),
                    uses_bias=use_bias,
                    num_batch_dims=num_batch_dims,
                )

    def test_qs8_linear_fused_relu(self):
        class LinearReluModule(torch.nn.Module):
            def __init__(self, in_size, out_size, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_size, out_size, bias=use_bias)

            def forward(self, x):
                return torch.nn.functional.relu(self.linear(x))

        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: LinearReluModule(
                        in_size,
                        out_size,
                        use_bias,  # noqa
                    ),
                    num_batch_dims=num_batch_dims,
                    uses_bias=use_bias,
                    quant=True,
                )

    def test_qs8_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    uses_bias=use_bias,
                    num_batch_dims=num_batch_dims,
                )

    @unittest.skip("XNNPACK currently only supports per-channel dynamic quantization.")
    def test_qd8_per_tensor_linear(self):
        for uses_bias in (False, True):
            inputs = (torch.randn(2, 4),)
            module = torch.nn.Linear(4, 5, bias=uses_bias)
            dynamic_shapes = ({0: torch.export.Dim("batch", max=100)},)

            self._test_dqlinear(
                module,
                inputs,
                dynamic_shapes=dynamic_shapes,
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
                dynamic_shapes=({0: torch.export.Dim("batch", max=100)},),
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
        batches = [2, 2]
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
                dynamic_shapes=({0: torch.export.Dim("batch", max=100)},),
                is_per_channel=True,
                uses_bias=bias,
                qconfig=qconfig,
            )

    def test_qd8_per_channel_linear_parallel(self):
        in_size = 2
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
        batch_dim = torch.export.Dim("batch", max=100)
        dynamic_shapes = ({0: batch_dim}, {0: batch_dim})

        self._test_dqlinear(
            ParallelLinear(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=2,
            is_per_channel=True,
            uses_bias=True,
        )

    def test_qd8_per_channel_linear_with_two_batch(self):
        in_size = 2
        input_size = 4
        output_size = 5

        linear = torch.nn.Linear(input_size, output_size)
        inputs = (torch.randn(2, in_size, input_size, dtype=torch.float),)
        batch_dim = torch.export.Dim("batch", max=100)
        dynamic_shapes = ({0: batch_dim, 1: batch_dim},)

        self._test_dqlinear(
            linear,
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=1,
            is_per_channel=True,
            uses_bias=True,
        )

    def test_qd8_per_channel_linear_sequential(self):
        in_size = 2
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
        dynamic_shapes = ({0: torch.export.Dim("batch", max=100)},)

        self._test_dqlinear(
            LinearSequential(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=2,
            is_per_channel=True,
            uses_bias=True,
            atol=1e-1,
        )

    def test_qd8_per_channel_linear_parellel_and_sequential(self):
        in_size = 2
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
        dynamic_shapes = (
            {0: torch.export.Dim("batch", max=100)},
            {0: torch.export.Dim("batch2", max=100)},
        )

        self._test_dqlinear(
            LinearModule(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            linear_count=3,
            is_per_channel=True,
            uses_bias=True,
            atol=1e-1,
        )

    class ManualDQLinear(torch.nn.Module):
        def __init__(
            self,
            input_channels: int = 4,
            output_channels: int = 4,
            dtype: torch.dtype = torch.float,
            weight_n_bit: int = 4,
            group_size: int = 0,
            force_groupwise_quant: bool = False,
            use_bias: bool = False,
        ):
            super().__init__()

            self.ic = input_channels
            self.oc = output_channels

            assert dtype in [torch.float, torch.half], "Unsupported op dtype"
            self.op_dtype = dtype

            self.group_size = self.ic if group_size == 0 else group_size
            self.num_groups = 1
            if self.group_size != self.ic:
                assert self.ic % self.group_size == 0
                assert self.group_size % 8 == 0  # TODO make this 16
                self.num_groups = self.ic // self.group_size

            assert weight_n_bit in [4, 8], "Unsupported weight_n_bit"
            self.w_n_bit = weight_n_bit
            self.w_quant_min, self.w_quant_max = self.get_min_max(self.w_n_bit)

            self.w = torch.nn.Parameter(
                torch.randn(self.oc, self.ic), requires_grad=False
            )
            self.w_q = torch.nn.Parameter(
                torch.zeros(self.oc, self.ic), requires_grad=False
            )
            # Quantize the weights as per folded setup
            if self.group_size != self.ic or force_groupwise_quant:
                self.w_scales = torch.nn.Parameter(
                    torch.zeros(self.oc, self.num_groups), requires_grad=False
                )
                self.w_zero_points = torch.nn.Parameter(
                    torch.zeros(self.oc, self.num_groups), requires_grad=False
                )
                self.quant_weight_per_channel_group()
            else:  # per_channel quantization
                self.w_scales = torch.nn.Parameter(
                    torch.zeros(self.oc), requires_grad=False
                )
                self.w_zero_points = torch.nn.Parameter(
                    torch.zeros(self.oc), requires_grad=False
                )
                self.quant_weight_per_channel()

            # TODO - change bias dtyoe to arg.dtype
            self.bias = (
                torch.nn.Parameter(torch.randn(self.oc), requires_grad=False)
                if use_bias
                else None
            )

        def get_min_max(self, n_bit: int = 4):
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            return min_int, max_int

        def get_channel_qparams_symmetric(
            self,
            w: torch.Tensor,
            n_bit: int = 4,
            precision: torch.dtype = torch.float32,
        ):
            assert w.dim() == 2

            to_quant = w.to(precision)
            assert torch.isnan(to_quant).sum() == 0

            max_val = to_quant.amax(dim=1, keepdim=True)
            min_val = to_quant.amin(dim=1, keepdim=True)
            min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
            max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

            min_int, max_int = self.get_min_max(n_bit)

            max_val_abs = torch.max(-min_val_neg, max_val_pos)
            scales = max_val_abs / (float(max_int - min_int) / 2)
            scales = torch.max(
                scales, torch.full_like(scales, torch.finfo(torch.float32).eps)
            )
            zeros = torch.full_like(scales, 0)
            return scales.to(precision).reshape(w.shape[0]), zeros.to(
                precision
            ).reshape(w.shape[0]).reshape(w.shape[0])

        # Note: not using from torchao.quantization.quant_primitives because it will run into op registraion issues
        def get_group_qparams_symmetric(
            self, w, n_bit=4, groupsize=128, precision=torch.float32
        ):
            # needed for GPTQ with padding
            if groupsize > w.shape[-1]:
                groupsize = w.shape[-1]
            assert groupsize > 1
            assert w.shape[-1] % groupsize == 0
            assert w.dim() == 2

            to_quant = w.reshape(-1, groupsize)
            assert torch.isnan(to_quant).sum() == 0

            max_val = to_quant.amax(dim=1, keepdim=True)
            min_val = to_quant.amin(dim=1, keepdim=True)
            min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
            max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

            max_val_abs = torch.max(-min_val_neg, max_val_pos)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))

            scales = max_val_abs / (float(max_int - min_int) / 2)
            scales = torch.max(
                scales, torch.full_like(scales, torch.finfo(torch.float32).eps)
            )
            # TODO: make sure abs(scales) is not too small?
            zeros = torch.full_like(scales, 0)
            return scales.to(precision).reshape(w.shape[0], -1), zeros.to(
                precision
            ).reshape(w.shape[0], -1)

        # Note: not using from torchao.quantization.quant_primitives because it will run into op registraion issues
        def group_quantize_tensor_symmetric(
            self, w, n_bit=4, group_size=128, precision=torch.float32
        ):
            scales, zeros = self.get_group_qparams_symmetric(
                w, n_bit, group_size, precision
            )
            n_bit = 4
            max_int = 2 ** (n_bit - 1) - 1
            min_int = -(2 ** (n_bit - 1))
            # TODO: currently we don't know how to express torch.int4, we'll
            # add torch.int4 to core later
            w_int8 = torch.ops.quantized_decomposed.quantize_per_channel_group(
                w, scales, zeros, min_int, max_int, torch.int8, group_size
            )

            return w_int8, scales, zeros

        def fwd_input_per_token(self, input: torch.Tensor) -> torch.Tensor:
            ip_quant_min = -128
            ip_quant_max = 127
            input = input.to(self.op_dtype)
            (
                ip_scales,
                ip_zero_points,
            ) = torch.ops.quantized_decomposed.choose_qparams_per_token_asymmetric(
                input, torch.int8
            )

            input = torch.ops.quantized_decomposed.quantize_per_token(
                input,
                ip_scales,
                ip_zero_points,
                ip_quant_min,
                ip_quant_max,
                torch.int8,
            )
            input = torch.ops.quantized_decomposed.dequantize_per_token(
                input,
                ip_scales,
                ip_zero_points,
                ip_quant_min,
                ip_quant_max,
                torch.int8,
                self.op_dtype,
            )
            input = input.to(self.op_dtype)
            return input

        def quant_weight_per_channel(self):
            (
                self.w_scales.data,
                self.w_zero_points.data,
            ) = self.get_channel_qparams_symmetric(
                self.w, n_bit=self.w_n_bit, precision=self.op_dtype
            )
            self.w_q.data = torch.ops.quantized_decomposed.quantize_per_channel(
                self.w,
                self.w_scales,
                self.w_zero_points,
                axis=0,
                quant_min=self.w_quant_min,
                quant_max=self.w_quant_max,
                dtype=torch.int8,
            )

        def quant_weight_per_channel_group(self):
            self.w_q.data, w, zp = self.group_quantize_tensor_symmetric(
                self.w,
                n_bit=self.w_n_bit,
                group_size=self.group_size,
            )
            expected_min, expected_max = self.get_min_max(self.w_n_bit)
            assert (
                torch.min(self.w_q.data) >= expected_min
            ), "Found smaller than min element in quantized weight tensor"
            assert (
                torch.max(self.w_q.data) <= expected_max
            ), "Found larger than max element in quantized weight tensor"
            assert (
                w.ndim == 2 and zp.ndim == 2
            ), f"Expecting 2d scales and zp tensors, but got {w.shape}, {zp.shape}"
            self.w_scales.data, self.w_zero_points.data = w, zp

        def fwd_weight_per_channel(self) -> torch.Tensor:
            # This is HACKY because the dequant will produce fp32
            return torch.ops.quantized_decomposed.dequantize_per_channel(
                self.w_q,
                self.w_scales,
                self.w_zero_points,
                axis=0,
                quant_min=self.w_quant_min,
                quant_max=self.w_quant_max,
                dtype=torch.int8,  # Regardless of w_n_bit, convert to 4b later
            )

        def fwd_weight_per_channel_group(self) -> torch.Tensor:
            return torch.ops.quantized_decomposed.dequantize_per_channel_group(
                self.w_q,
                self.w_scales,
                self.w_zero_points,
                self.w_quant_min,
                self.w_quant_max,
                dtype=torch.int8,  # Regardless of w_n_bit, convert to 4b later
                group_size=self.group_size,
                output_dtype=self.op_dtype,
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Input
            input = self.fwd_input_per_token(input)

            # Weights
            w = (
                self.fwd_weight_per_channel_group()
                if self.w_scales.ndim == 2
                else self.fwd_weight_per_channel()
            )
            assert isinstance(w, torch.Tensor)
            return torch.nn.functional.linear(input, w, self.bias)

    def _test_manual_dq_linear(
        self,
        mod: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        weight_groupwise: bool = False,
        use_bias: bool = False,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ):
        linear_edge_op = (
            "executorch_exir_dialects_edge__ops_aten_addmm_default"
            if use_bias
            else "executorch_exir_dialects_edge__ops_aten_mm_default"
        )

        weight_dq_edge_op = (
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_group_default"
            if weight_groupwise
            else "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default"
        )

        (
            Tester(mod, inputs)
            .export()
            .to_edge()
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_per_token_asymmetric_default": 1,
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_token_default": 1,
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_token_default": 1,
                    weight_dq_edge_op: 1,
                    linear_edge_op: 1,
                }
            )
            .partition(Partition(partitioner=XnnpackDynamicallyQuantizedPartitioner()))
            .check_count(
                {
                    "torch.ops.higher_order.executorch_call_delegate": 1,
                }
            )
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_choose_qparams_per_token_asymmetric_default",
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_token_default",
                    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_token_default",
                    weight_dq_edge_op,
                    linear_edge_op,
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(atol=atol, rtol=rtol)
        )

    def _run_manual_dqlinear_tests(self, weight_n_bit: int, op_dtype: torch.dtype):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]

        for use_bias in [True, False]:
            for i, _ in enumerate(in_sizes):
                in_size = int(in_sizes[i])
                input_size = int(input_sizes[i])
                output_size = int(output_sizes[i])
                mod = self.ManualDQLinear(
                    input_channels=input_size,
                    output_channels=output_size,
                    weight_n_bit=weight_n_bit,
                    dtype=op_dtype,
                    use_bias=use_bias,
                )

                inputs = (torch.randn(1, in_size, input_size).to(op_dtype),)
                self._test_manual_dq_linear(mod, inputs, use_bias=use_bias)

    def test_qd8_fp32_per_token_weight_per_channel_int8(self):
        self._run_manual_dqlinear_tests(8, torch.float)

    def test_qd8_fp32_per_token_weight_per_channel_int4(self):
        self._run_manual_dqlinear_tests(4, torch.float)

    # This fails because the output tensor dtype is different, but if you squint and ignore that and look at the values,
    # it is not too bad.
    # Difference: max: 0.042601585388183594, abs: 0.042601585388183594.
    # -- Model vs. Reference --
    #  Numel: 68, 68
    # Median: -0.7754800915718079, -0.7755751013755798
    #   Mean: -0.6128872036933899, -0.6143574714660645
    #    Max: 12.518657684326172, 12.516003608703613
    #    Min: -20.070953369140625, -20.077701568603516
    @unittest.skip("Need to fix the dq_per_channel output dtype")
    def test_qd8_fp16_per_token_weight_per_channel_int8(self):
        self._run_manual_dqlinear_tests(8, torch.float16)

    @unittest.skip("Need to fix the dq_per_channel output dtype")
    def test_qd8_fp16_per_token_weight_per_channel_int4(self):
        self._run_manual_dqlinear_tests(4, torch.float16)

    def test_qd8_fp32_per_token_weight_per_channel_group_int4(self):
        M_sizes = [1, 2, 17, 31]
        K_sizes = [8, 32, 64, 128]
        bl_sizes = [8, 16, 16, 32]
        N_sizes = [2, 17, 92, 128]

        for use_bias in [True, False]:
            for i, _ in enumerate(M_sizes):
                M = int(M_sizes[i])
                K = int(K_sizes[i])
                N = int(N_sizes[i])
                bl = int(bl_sizes[i])
                mod = self.ManualDQLinear(
                    input_channels=K,
                    output_channels=N,
                    weight_n_bit=4,
                    dtype=torch.float,
                    group_size=bl,
                    force_groupwise_quant=True,
                    use_bias=use_bias,
                )

                inputs = (torch.randn(1, M, K),)
                self._test_manual_dq_linear(
                    mod,
                    inputs,
                    weight_groupwise=True,
                    use_bias=use_bias,
                )

    def _test_linear(
        self,
        make_module,
        uses_bias,
        num_batch_dims=1,
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

        in_sizes = [3, 4, 4]
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
            input_shape = [in_size] * num_batch_dims + [input_size]
            print(f"Testing input_shape {input_shape} with {output_size} out_channels")

            module = make_module(input_size, output_size).eval().to(dtype)
            inputs = (torch.randn(input_shape).to(dtype),)
            dynamic_shape = {}
            for i in range(num_batch_dims):
                dynamic_shape[i] = torch.export.Dim(f"batch{i}", min=2, max=in_size)

            dynamic_shape = (dynamic_shape,)
            print(dynamic_shape)

            tester = Tester(module, inputs, dynamic_shapes=dynamic_shape)

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
            tester.run_method_and_compare_outputs(qtol=quant, atol=atol)

    def _test_dqlinear(
        self,
        module,
        inputs,
        dynamic_shapes,
        linear_count=1,
        is_per_channel=False,
        uses_bias=False,
        qconfig: Optional[QuantizationConfig] = None,
        atol=5e-02,
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

        tester = Tester(module, inputs, dynamic_shapes=dynamic_shapes)
        tester.quantize(Quantize(quantization_config=quant_config))

        tester.export()
        tester.check_count({aten_op: linear_count})
        tester.check(["torch.ops.quantized_decomposed"])
        tester.to_edge()
        tester.check_count({edge_op: linear_count})

        tester.partition(
            Partition(partitioner=XnnpackDynamicallyQuantizedPartitioner())
        )
        tester.check(["torch.ops.higher_order.executorch_call_delegate"])
        tester.check_not([edge_op])

        tester.to_executorch()
        tester.serialize()
        tester.run_method_and_compare_outputs(atol=atol)
