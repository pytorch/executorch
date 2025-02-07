#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import inspect

import unittest

from typing import Tuple

import torch
from executorch.backends.apple.mps.test.test_mps_utils import TestMPS


class TestLinear(TestMPS):
    @unittest.skip("Dynamic shapes not supported in MPS backend")
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

    @unittest.skip("Dynamic shapes not supported in MPS backend")
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

    @unittest.skip("Dynamic shapes not supported in MPS backend")
    def test_qc8_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    uses_bias=use_bias,
                    quant_type="per_channel",
                    num_batch_dims=num_batch_dims,
                )

    @unittest.skip("Dynamic shapes not supported in MPS backend")
    def test_fp32_addmm(self):
        """
        Note that the ConvertToLinear pass requires the weight matrix to be transposed.
        """

        class AddMMModule(torch.nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()
                self.mat = torch.nn.Parameter(torch.randn(in_size, out_size))
                self.bias = torch.nn.Parameter(torch.randn(1, out_size))

            def forward(self, x):
                return torch.addmm(self.bias, x, self.mat)

        self._test_linear(
            lambda in_size, out_size: AddMMModule(in_size, out_size),
            uses_bias=True,
        )

    @unittest.skip("Dynamic shapes not supported in MPS backend")
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

    @unittest.skip("Dynamic shapes not supported in MPS backend")
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
                    quant_type="per_tensor",
                )

    @unittest.skip("Dynamic shapes not supported in MPS backend")
    def test_qs8_linear(self):
        for use_bias in (True, False):
            for num_batch_dims in range(1, 3):
                self._test_linear(
                    lambda in_size, out_size: torch.nn.Linear(
                        in_size, out_size, bias=use_bias  # noqa
                    ),
                    uses_bias=use_bias,
                    num_batch_dims=num_batch_dims,
                    quant_type="per_tensor",
                )

    @unittest.skip(
        "quantized_decomposed_dequantize_per_channel_default is not supported bt MPS delegate"
    )
    def test_qd8_fp32_per_token_weight_per_channel_int8(self):
        self._run_manual_dqlinear_tests(8, torch.float)

    @unittest.skip(
        "quantized_decomposed_dequantize_per_channel_default is not supported bt MPS delegate"
    )
    def test_qd8_fp32_per_token_weight_per_channel_int4(self):
        self._run_manual_dqlinear_tests(4, torch.float)

    def test_qd8_fp32_per_token_weight_per_channel_group_int4(self):
        M_sizes = [1]
        K_sizes = [64]
        bl_sizes = [64]
        N_sizes = [32]

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

    @unittest.skip("Need to fix the dq_per_channel_group output dtype")
    def _test_qd8_fp16_per_token_weight_per_channel_group_int4(self):
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
                    dtype=torch.float16,
                    group_size=bl,
                    force_groupwise_quant=True,
                    use_bias=use_bias,
                )

                inputs = (torch.randn(1, M, K, dtype=torch.float16),)
                self._test_manual_dq_linear(
                    mod,
                    inputs,
                    weight_groupwise=True,
                    use_bias=use_bias,
                    atol=0.1,
                    rtol=0.1,
                )

    def _test_linear(
        self,
        make_module,
        uses_bias,
        num_batch_dims=1,
        quant_type=None,
        dtype: torch.dtype = torch.float,
        atol=1e-03,
    ):
        in_sizes = [3, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]

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
            self.lower_and_test_without_partitioner(
                module,
                inputs,
                func_name=inspect.stack()[0].function[5:],
                dynamic_shapes=dynamic_shape,
                atol=atol,
                rtol=1e-03,
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

            self.bias = (
                torch.nn.Parameter(
                    torch.randn(self.oc).to(self.op_dtype), requires_grad=False
                )
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
    ):
        self.lower_and_test_without_partitioner(
            mod, inputs, func_name=inspect.stack()[0].function[5:]
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
