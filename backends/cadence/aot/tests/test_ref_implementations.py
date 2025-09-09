# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import typing
import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.cadence.aot.ref_implementations  # noqa

import numpy as np
import torch
from executorch.backends.cadence.aot.typing_stubs import expand


class TestRefImplementations(unittest.TestCase):
    @expand(
        [
            ("basic_int8", 0.42, -1.0, 2.0, -7, 7, torch.int8, 0),
            ("basic_int16", 0.42, -1.0, 5.0, -6, 7, torch.int16, -3),
        ]
    )
    def test_quantize_per_tensor(
        self,
        name: str,
        input_value: float,
        f_min: float,
        f_max: float,
        q_min: int,
        q_max: int,
        target_dtype: torch.dtype,
        expected_value: int,
    ) -> None:
        input_tensor = torch.tensor([input_value])
        scale = (f_max - f_min) / (q_max - q_min)
        inv_scale = 1.0 / scale
        zero_point = round(-f_min * inv_scale) + q_min
        expected_output = torch.tensor([expected_value], dtype=target_dtype)

        output = torch.ops.cadence.quantize_per_tensor(
            input_tensor, inv_scale, zero_point, q_min, q_max, target_dtype
        )

        self.assertEqual(
            output.dtype, expected_output.dtype, f"Dtype mismatch in {name}"
        )
        self.assertTrue(
            torch.equal(output, expected_output),
            f"Values don't match in {name}: got {output}, expected {expected_output}",
        )

    @expand(
        [
            # Signed quantization ranges
            ("signed_range0_int8", 0, -1.0, 2.0, -7, 7, torch.int8, 0.428),
            ("signed_range0_int16", 0, -1.0, 2.0, -7, 7, torch.int16, 0.428),
            ("signed_range0_int32", 0, -1.0, 2.0, -7, 7, torch.int32, 0.428),
            ("signed_range1_int8", -3, -1.0, 5.0, -6, 7, torch.int8, 0.461),
            ("signed_range1_int16", -3, -1.0, 5.0, -6, 7, torch.int16, 0.461),
            ("signed_range1_int32", -3, -1.0, 5.0, -6, 7, torch.int32, 0.461),
            # Unsigned quantization ranges
            ("unsigned_range0_uint8", 3, -1.0, 2.0, 0, 7, torch.uint8, 0.428),
            ("unsigned_range0_uint16", 3, -1.0, 2.0, 0, 7, torch.uint16, 0.428),
            ("unsigned_range1_uint8", 4, -1.0, 5.0, 3, 7, torch.uint8, 0.0),
            ("unsigned_range1_uint16", 4, -1.0, 5.0, 3, 7, torch.uint16, 0.0),
        ]
    )
    def test_dequantize_per_tensor(
        self,
        name: str,
        input_value: int,
        f_min: float,
        f_max: float,
        q_min: int,
        q_max: int,
        input_dtype: torch.dtype,
        expected_value: int,
    ) -> None:
        input_tensor = torch.tensor([input_value], dtype=input_dtype)
        scale = (f_max - f_min) / (q_max - q_min)
        zero_point = round(-f_min / scale) + q_min
        expected_output = torch.tensor([expected_value], dtype=torch.float32)

        output = torch.ops.cadence.dequantize_per_tensor(
            input_tensor, scale, zero_point, q_min, q_max, torch.float32
        )

        self.assertEqual(
            output.dtype, expected_output.dtype, f"Dtype mismatch in {name}"
        )
        self.assertTrue(
            torch.allclose(output, expected_output, rtol=0.001, atol=0.001),
            f"Values don't match in {name}: got {output}, expected {expected_output}",
        )

    @expand(
        [
            # Only these types need to be tested as per ET_FORALL_JARVIS_QUANTIZED_TYPES in
            # on_device_ai/Assistant/Jarvis/min_runtime/operators/generic/operators.h
            ("int8", 5, 0.8, 4, 5, 0.8, 4, 0.8, 4, 6, torch.int8),
            ("uint8", 5, 0.8, 4, 5, 0.8, 4, 0.8, 4, 6, torch.uint8),
        ]
    )
    def test_quantized_add(
        self,
        name: str,
        X: int,
        X_scale: float,
        X_zero_point: int,
        Y: int,
        Y_scale: float,
        Y_zero_point: int,
        out_scale: float,
        out_zero_point: int,
        expected_value: int,
        dtype: torch.dtype,
    ) -> None:
        X_tensor = torch.tensor([X], dtype=dtype)
        Y_tensor = torch.tensor([Y], dtype=dtype)
        expected_output = torch.tensor([expected_value], dtype=dtype)

        quantized_add = (
            torch.ops.cadence.quantized_add_asym8sxasym8s_asym8s.per_tensor
            if dtype == torch.int8
            else torch.ops.cadence.quantized_add_asym8uxasym8u_asym8u.per_tensor
        )
        output = quantized_add(
            X_tensor,
            X_scale,
            X_zero_point,
            Y_tensor,
            Y_scale,
            Y_zero_point,
            out_scale,
            out_zero_point,
        )

        self.assertTrue(
            torch.equal(output, expected_output),
            f"Values don't match in {name}: got {output}, expected {expected_output}",
        )

        output = torch.ops.cadence.quantized_add(
            X_tensor,
            X_scale,
            X_zero_point,
            Y_tensor,
            Y_scale,
            Y_zero_point,
            out_scale,
            out_zero_point,
        )

        self.assertTrue(
            torch.equal(output, expected_output),
            f"Values don't match in {name}: got {output}, expected {expected_output}",
        )

    @expand(
        [
            # Test case 1: 1x2 input, 1x2 weight (1 output feature)
            *[
                (
                    torch.Size([1, 2]),  # src_shape: 1 sample, 2 input features
                    torch.Size(
                        [1, 2]
                    ),  # weight_shape: 1 output feature, 2 input features
                    0,  # in_zero_point
                    torch.tensor([0, 0], dtype=dtype),  # weight_zero_point
                    torch.tensor(
                        [1073741824], dtype=torch.int32
                    ),  # out_multiplier (0.5 * 2^31)
                    torch.tensor([0], dtype=torch.int64),  # out_shift
                    0,  # out_zero_point
                    torch.tensor([[-2]], dtype=dtype),  # expected_output
                    per_tensor,
                    False,
                    False,
                )
                for (per_tensor, dtype) in (
                    (False, torch.int8),
                    (True, torch.int8),
                    (True, torch.uint8),
                )
            ],
            # Test case 2: 1x3 input, 2x3 weight (2 output features)
            *[
                (
                    torch.Size([1, 3]),  # src_shape: 1 sample, 3 input features
                    torch.Size(
                        [2, 3]
                    ),  # weight_shape: 2 output features, 3 input features
                    0,  # in_zero_point
                    torch.tensor([0, 0, 0], dtype=dtype),  # weight_zero_point
                    torch.tensor(
                        [1073741824], dtype=torch.int32
                    ),  # out_multiplier (0.5 * 2^31)
                    torch.tensor([0], dtype=torch.int64),  # out_shift
                    0,  # out_zero_point
                    torch.tensor([[-10, -30]], dtype=dtype),  # expected_output
                    per_tensor,
                    False,
                    False,
                )
                for (per_tensor, dtype) in (
                    (False, torch.int8),
                    (True, torch.int8),
                    (True, torch.uint8),
                )
            ],
            # Test case 3: Batch case with different dimensions
            *[
                (
                    torch.Size([1, 2, 2]),  # src_shape: batch=1, seq=2, features=2
                    torch.Size(
                        [3, 2]
                    ),  # weight_shape: 3 output features, 2 input features
                    0,  # in_zero_point
                    torch.tensor([0, 0], dtype=dtype),  # weight_zero_point
                    torch.tensor(
                        [1073741824], dtype=torch.int32
                    ),  # out_multiplier (0.5 * 2^31)
                    torch.tensor([0], dtype=torch.int64),  # out_shift
                    0,  # out_zero_point
                    torch.tensor(
                        [[[-2, -8, -14], [-6, -28, -50]]], dtype=dtype
                    ),  # expected_output
                    per_tensor,
                    False,
                    False,
                )
                for (per_tensor, dtype) in (
                    (False, torch.int8),
                    (True, torch.int8),
                    (True, torch.uint8),
                )
            ],
            # Test case 4: Non-zero zero points
            *[
                (
                    torch.Size([1, 2]),  # src_shape: 1 sample, 2 input features
                    torch.Size(
                        [2, 2]
                    ),  # weight_shape: 2 output feature, 1 input feature
                    2,  # in_zero_point
                    torch.tensor([1, 1], dtype=dtype),  # weight_zero_point
                    torch.tensor(
                        [268435456], dtype=torch.int32
                    ),  # out_multiplier (1.0 * 2^31)
                    torch.tensor([0], dtype=torch.int64),  # out_shift
                    1,  # out_zero_point
                    torch.tensor([[-15, 25]], dtype=dtype),  # expected_output
                    per_tensor,
                    False,
                    False,
                )
                for (per_tensor, dtype) in (
                    (False, torch.int8),
                    (True, torch.int8),
                    (True, torch.uint8),
                )
            ],
            # Test case 5: Non-uniform weight zero points
            *[
                (
                    torch.Size([1, 2]),  # src_shape: 1 sample, 2 input features
                    torch.Size(
                        [2, 2]
                    ),  # weight_shape: 2 output feature, 1 input feature
                    2,  # in_zero_point
                    torch.tensor([1, 2], dtype=dtype),  # weight_zero_point
                    torch.tensor(
                        [268435456], dtype=torch.int32
                    ),  # out_multiplier (1.0 * 2^31)
                    torch.tensor([0], dtype=torch.int64),  # out_shift
                    1,  # out_zero_point
                    torch.tensor([[-23, 17]], dtype=dtype),  # expected_output
                    False,
                    False,
                    False,
                )
                for dtype in (torch.int8, torch.uint8)
            ],
            # Test case 6: Non-zero out_shift (shift=1)
            *[
                (
                    torch.Size([1, 2]),  # src_shape: 1 sample, 2 input features
                    torch.Size(
                        [2, 2]
                    ),  # weight_shape: 2 output features, 2 input features
                    2,  # in_zero_point
                    torch.tensor([1, 1], dtype=dtype),  # weight_zero_point
                    torch.tensor(
                        [268435456], dtype=torch.int32
                    ),  # out_multiplier (0.125 * 2^31)
                    torch.tensor(
                        [1], dtype=torch.int64
                    ),  # out_shift (shift=1, doubles the scale)
                    1,  # out_zero_point
                    torch.tensor([[-7, 13]], dtype=dtype),  # expected_output
                    per_tensor,
                    False,
                    False,
                )
                for (per_tensor, dtype) in ((False, torch.int8), (True, torch.int8))
            ],
            *[
                (
                    torch.Size([1, 2]),  # src_shape: 1 sample, 2 input features
                    torch.Size(
                        [2, 2]
                    ),  # weight_shape: 2 output features, 2 input features
                    2,  # in_zero_point
                    torch.tensor([1, 1], dtype=dtype),  # weight_zero_point
                    torch.tensor(
                        [268435456], dtype=torch.int32
                    ),  # out_multiplier (0.125 * 2^31)
                    torch.tensor(
                        [1], dtype=torch.int64
                    ),  # out_shift (shift=1, doubles the scale)
                    1,  # out_zero_point
                    torch.tensor([[-7, 17]], dtype=dtype),  # expected_output
                    per_tensor,
                    matmul,
                    transposed_matmul,
                )
                for (matmul, transposed_matmul) in ((True, False), (True, True))
                for (per_tensor, dtype) in ((True, torch.int8), (True, torch.uint8))
            ],
        ]
    )
    def test_quantized_linear(
        self,
        src_shape: torch.Size,
        weight_shape: torch.Size,
        in_zero_point: int,
        weight_zero_point: torch.Tensor,
        out_multiplier: torch.Tensor,
        out_shift: torch.Tensor,
        out_zero_point: int,
        expected_output: torch.Tensor,
        per_tensor: bool,
        matmul: bool,
        transposed_matmul: bool,
    ) -> None:
        if not per_tensor and matmul:
            self.skipTest("Only per_tensor supported for matmul")

        src = (
            torch.arange(np.prod(src_shape))
            .reshape(src_shape)
            .to(expected_output.dtype)
        )
        weight = (
            torch.arange(np.prod(weight_shape))
            .reshape(weight_shape)
            .to(expected_output.dtype)
        )
        if matmul and not transposed_matmul:
            weight = weight.T

        if per_tensor:
            weight_zero_point = weight_zero_point[0]
            out_multiplier = out_multiplier[0]
            out_shift = out_shift[0]

        if per_tensor:
            match expected_output.dtype:
                case torch.int8:
                    if matmul:
                        linear_ops = (
                            # Doesn't have per tensor name, but it is per tensor
                            torch.ops.cadence.quantized_matmul_asym8sxasym8s_asym8s,
                        )
                    else:
                        linear_ops = (
                            torch.ops.cadence.quantized_linear_asym8sxasym8s_asym8s.per_tensor,
                            torch.ops.cadence.quantized_fully_connected_asym8sxasym8s_asym8s.per_tensor,
                        )
                case torch.uint8:
                    if matmul:
                        linear_ops = (
                            torch.ops.cadence.quantized_matmul_asym8uxasym8u_asym8u,
                        )
                    else:
                        linear_ops = (
                            torch.ops.cadence.quantized_linear_asym8uxasym8u_asym8u.per_tensor,
                            torch.ops.cadence.quantized_fully_connected_asym8uxasym8u_asym8u.per_tensor,
                        )
                case _:
                    if matmul:
                        linear_ops = (torch.ops.cadence.quantized_matmul,)
                    else:
                        linear_ops = (
                            torch.ops.cadence.quantized_linear.per_tensor,
                            torch.ops.cadence.quantized_fully_connected.per_tensor,
                        )
        else:
            linear_ops = (
                torch.ops.cadence.quantized_linear,
                torch.ops.cadence.quantized_fully_connected,
            )

        for linear_op in linear_ops:
            # Get the function name for linear_op for debugging
            op_name = (
                linear_op.__name__ if hasattr(linear_op, "__name__") else str(linear_op)
            )
            if matmul:
                assert "quantized_matmul" in op_name
                output = linear_op(
                    src,
                    in_zero_point,
                    weight,
                    weight_zero_point,
                    None,
                    out_multiplier,
                    out_shift,
                    out_zero_point,
                    transposed_matmul,
                )
            else:
                assert (
                    "quantized_linear" in op_name
                    or "quantized_fully_connected" in op_name
                )
                bias = torch.arange(weight_shape[0]).to(torch.int32)
                output = linear_op(
                    src,
                    weight,
                    bias,
                    in_zero_point,
                    weight_zero_point,
                    out_multiplier,
                    out_shift,
                    out_zero_point,
                    typing.cast(torch.Tensor, None),
                )

            self.assertTrue(output.dtype == expected_output.dtype, "Dtype mismatch")

            self.assertTrue(
                torch.equal(output, expected_output),
                f"Values don't match: got {output}, expected {expected_output}",
            )

    @expand(
        [
            # Test case 1: Simple case with int8, zero mean input
            (
                torch.tensor(
                    [[-1, 1]], dtype=torch.int8
                ),  # input: dequantized to [-0.1, 0.1]
                0.1,  # X_scale
                0,  # X_zero_point
                [2],  # normalized_shape (last dimension)
                torch.tensor([1.0, 1.0]),  # weight
                torch.tensor([0.0, 0.0]),  # bias
                1e-5,  # eps
                0.1,  # output_scale
                0,  # output_zero_point
                torch.int8,  # dtype
                torch.tensor([[-10, 10]], dtype=torch.int8),  # expected_output
            ),
            # Test case 2: uint8 with zero_point offset
            (
                torch.tensor(
                    [[127, 129]], dtype=torch.uint8
                ),  # input: dequantized to [-0.05, 0.05]
                0.05,  # X_scale
                128,  # X_zero_point
                [2],  # normalized_shape (last dimension)
                torch.tensor([1.0, 1.0]),  # weight
                torch.tensor([0.0, 0.0]),  # bias
                1e-5,  # eps
                0.05,  # output_scale
                128,  # output_zero_point
                torch.uint8,  # dtype
                torch.tensor([[108, 148]], dtype=torch.uint8),  # expected_output
            ),
            # Test case 3: Test with weight and bias scaling
            (
                torch.tensor(
                    [[-2, 2]], dtype=torch.int8
                ),  # input: dequantized to [-0.2, 0.2]
                0.1,  # X_scale
                0,  # X_zero_point
                [2],  # normalized_shape (last dimension)
                torch.tensor(
                    [2.0, 0.5]
                ),  # weight: scale first element by 2, second by 0.5
                torch.tensor(
                    [0.1, -0.1]
                ),  # bias: add 0.1 to first, subtract 0.1 from second
                1e-5,  # eps
                0.1,  # output_scale
                0,  # output_zero_point
                torch.int8,  # dtype
                torch.tensor([[-19, 4]], dtype=torch.int8),  # expected_output
            ),
        ]
    )
    def test_quantized_layer_norm_per_tensor(
        self,
        input_tensor: torch.Tensor,
        X_scale: float,
        X_zero_point: int,
        normalized_shape: list[int],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        output_scale: float,
        output_zero_point: int,
        dtype: torch.dtype,
        expected_output: torch.Tensor,
    ) -> None:
        output = torch.ops.cadence.quantized_layer_norm.per_tensor(
            input_tensor,
            X_scale,
            X_zero_point,
            normalized_shape,
            weight,
            bias,
            eps,
            output_scale,
            output_zero_point,
        )

        # Verify output properties
        self.assertEqual(output.dtype, dtype, f"Output dtype should be {dtype}")
        self.assertEqual(
            output.shape, input_tensor.shape, "Output shape should match input shape"
        )

        # Verify output matches expected values
        self.assertTrue(
            torch.equal(output, expected_output),
            f"Output values don't match expected. Got {output}, expected {expected_output}",
        )

    @expand(
        [
            # Test case 1: Basic 2D convolution with int8
            *[
                (
                    torch.tensor(
                        [[[[1, 2], [3, 4]]]], dtype=torch.int8
                    ),  # input: 1x1x2x2
                    torch.tensor(
                        [[[[1, 0], [0, 1]]]], dtype=torch.int8
                    ),  # weight: 1x1x2x2 (identity-like)
                    torch.tensor([0], dtype=torch.int32),  # bias
                    (1, 1),  # stride
                    (0, 0),  # padding
                    (1, 1),  # dilation
                    1,  # groups
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.1,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int8,  # dtype
                    torch.tensor(
                        [[[[50]]]], dtype=torch.int8
                    ),  # expected_output: (1*1 + 4*1) / 0.1 = 50
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
            # Test case 2: 2D convolution with stride and padding
            *[
                (
                    torch.tensor(
                        [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.int8
                    ),  # input: 1x1x3x3
                    torch.tensor(
                        [[[[1, 1], [1, 1]]]], dtype=torch.int8
                    ),  # weight: 1x1x2x2 (sum filter)
                    torch.tensor([0], dtype=torch.int32),  # bias
                    (1, 1),  # stride
                    (0, 0),  # padding
                    (1, 1),  # dilation
                    1,  # groups
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.25,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int8,  # dtype
                    torch.tensor(
                        [[[[48, 64], [96, 112]]]], dtype=torch.int8
                    ),  # expected_output: convolution results with output_scale=0.25
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
            # Test case 3: uint8 with non-zero zero points
            *[
                (
                    torch.tensor(
                        [[[[130, 132], [134, 136]]]], dtype=torch.uint8
                    ),  # input: 1x1x2x2
                    torch.tensor(
                        [[[[129, 128], [128, 129]]]], dtype=torch.uint8
                    ),  # weight: 1x1x2x2 (values close to zero_point)
                    torch.tensor([10], dtype=torch.int32),  # bias
                    (1, 1),  # stride
                    (0, 0),  # padding
                    (1, 1),  # dilation
                    1,  # groups
                    128,  # in_zero_point
                    128,  # weight_zero_point
                    0.1,  # bias_scale
                    0.1,  # output_scale
                    128,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.uint8,  # dtype
                    torch.tensor(
                        [[[[238]]]], dtype=torch.uint8
                    ),  # (130 - 128) + (134 - 128) = 10
                    # + bias -> 10 + 1 = 11
                    # round(11 / 0.1 + 128) = 238,
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
            # Test case 4: 1D convolution (3D input tensor)
            *[
                (
                    torch.tensor(
                        [[[1, 2, 3, 4]]], dtype=torch.int8
                    ),  # input: 1x1x4 (N, C, W)
                    torch.tensor(
                        [[[1, 1]]], dtype=torch.int8
                    ),  # weight: 1x1x2 (OC, IC, KW)
                    torch.tensor([0], dtype=torch.int32),  # bias
                    (1, 1),  # stride (padding for 2D, actual stride is stride[1])
                    (0, 0),  # padding (padding for 2D, actual padding is padding[1])
                    (1, 1),  # dilation (padding for 2D, actual dilation is dilation[1])
                    1,  # groups
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.5,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int8,  # dtype
                    torch.tensor(
                        [[[6, 10, 14]]], dtype=torch.int8
                    ),  # expected_output: [1+2, 2+3, 3+4] / 0.5 = [6, 10, 14]
                    memory_format,
                )
                for memory_format in [torch.contiguous_format]
            ],
            # Test case 5: Multiple output channels
            *[
                (
                    torch.tensor(
                        [[[[1, 2], [3, 4]]]], dtype=torch.int8
                    ),  # input: 1x1x2x2
                    torch.tensor(
                        [
                            [[[1, 0], [0, 1]]],  # first output channel
                            [[[0, 1], [1, 0]]],  # second output channel
                        ],
                        dtype=torch.int8,
                    ),  # weight: 2x1x2x2
                    torch.tensor(
                        [0, 5], dtype=torch.int32
                    ),  # bias for each output channel
                    (1, 1),  # stride
                    (0, 0),  # padding
                    (1, 1),  # dilation
                    1,  # groups
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.2,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int8,  # dtype
                    torch.tensor(
                        [[[[25]], [[50]]]], dtype=torch.int8
                    ),  # expected_output: [5/0.2, 10/0.2] = [25, 50]
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
            # Test case 6: Multiple input channels
            *[
                (
                    torch.tensor(
                        [
                            [
                                [[1, 2], [3, 4]],  # first input channel
                                [[5, 6], [7, 8]],
                            ]  # second input channel
                        ],
                        dtype=torch.int16,
                    ),  # input: 1x2x2x2
                    torch.tensor(
                        [
                            [
                                [[1, 0], [0, 1]],  # weights for first input channel
                                [[0, 1], [1, 0]],
                            ]  # weights for second input channel
                        ],
                        dtype=torch.int16,
                    ),  # weight: 1x2x2x2 (1 output channel, 2 input channels)
                    torch.tensor([0], dtype=torch.int32),  # bias
                    (1, 1),  # stride
                    (0, 0),  # padding
                    (1, 1),  # dilation
                    1,  # groups
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.1,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int16,  # dtype
                    torch.tensor(
                        [[[[180]]]], dtype=torch.int16
                    ),  # expected_output: (1 + 4 + 6 + 7) / 0.1 = 180
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
            # Test case 7: Multiple input and output channels
            *[
                (
                    torch.tensor(
                        [
                            [
                                [[1, 2], [3, 4]],  # first input channel
                                [[2, 1], [4, 3]],
                            ]  # second input channel
                        ],
                        dtype=torch.int16,
                    ),  # input: 1x2x2x2
                    torch.tensor(
                        [
                            [
                                [
                                    [1, 1],
                                    [1, 1],
                                ],  # first output channel, first input channel
                                [[1, 1], [1, 1]],
                            ],  # first output channel, second input channel
                            [
                                [
                                    [1, 0],
                                    [0, 1],
                                ],  # second output channel, first input channel
                                [[0, 1], [1, 0]],
                            ],  # second output channel, second input channel
                        ],
                        dtype=torch.int16,
                    ),  # weight: 2x2x2x2 (2 output channels, 2 input channels)
                    torch.tensor(
                        [0, 0], dtype=torch.int32
                    ),  # bias for each output channel
                    (1, 1),  # stride
                    (0, 0),  # padding
                    (1, 1),  # dilation
                    1,  # groups
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.05,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int16,  # dtype
                    torch.tensor([[[[400]], [[200]]]], dtype=torch.int16),
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
            # Test case 8: Grouped convolution (groups=2)
            *[
                (
                    torch.tensor(
                        [
                            [
                                [[1, 2], [3, 4]],  # first input channel (group 1)
                                [[5, 6], [7, 8]],
                            ]  # second input channel (group 2)
                        ],
                        dtype=torch.int8,
                    ),  # input: 1x2x2x2
                    torch.tensor(
                        [
                            [
                                [[1, 1], [1, 1]]
                            ],  # first output channel (processes first input channel)
                            [
                                [[1, 0], [0, 1]]
                            ],  # second output channel (processes second input channel)
                        ],
                        dtype=torch.int8,
                    ),  # weight: 2x1x2x2 (2 output channels, 1 input channel each due to groups=2)
                    torch.tensor(
                        [0, 0], dtype=torch.int32
                    ),  # bias for each output channel
                    (1, 1),  # stride
                    (0, 0),  # padding
                    (1, 1),  # dilation
                    2,  # groups (grouped convolution)
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.2,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int8,  # dtype
                    torch.tensor(
                        [[[[50]], [[65]]]], dtype=torch.int8
                    ),  # expected_output: [(1+2+3+4)/0.2, (5+8)/0.2] = [50, 65]
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
            # Test case 9: Convolution with stride=2 and padding=1
            *[
                (
                    torch.tensor(
                        [
                            [
                                [
                                    [1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16],
                                ]
                            ]
                        ],
                        dtype=torch.int8,
                    ),  # input: 1x1x4x4
                    torch.tensor(
                        [[[[1, 1], [1, 1]]]], dtype=torch.int8
                    ),  # weight: 1x1x2x2 (sum filter)
                    torch.tensor([0], dtype=torch.int32),  # bias
                    (2, 2),  # stride=2
                    (1, 1),  # padding=1
                    (1, 1),  # dilation
                    1,  # groups
                    0,  # in_zero_point
                    0,  # weight_zero_point
                    1.0,  # bias_scale
                    0.5,  # output_scale
                    0,  # output_zero_point
                    0,  # unused out_multiplier
                    0,  # unused out_shift
                    torch.int8,  # dtype
                    torch.tensor(
                        [[[[2, 10, 8], [28, 68, 40], [26, 58, 32]]]], dtype=torch.int8
                    ),
                    memory_format,
                )
                for memory_format in [torch.contiguous_format, torch.channels_last]
            ],
        ]
    )
    def test_quantized_conv_per_tensor(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        in_zero_point: int,
        weight_zero_point: int,
        bias_scale: float,
        output_scale: float,
        output_zero_point: int,
        out_multiplier: int,
        out_shift: int,
        dtype: torch.dtype,
        expected_output: torch.Tensor,
        memory_format: torch.memory_format,
    ) -> None:
        assert memory_format in [torch.contiguous_format, torch.channels_last]

        if len(input_tensor.shape) == 3 and memory_format == torch.channels_last:
            self.fail("Channels last format is not supported for 3D input tensors")

        if memory_format == torch.channels_last:
            input_tensor = torch.permute(input_tensor, (0, 2, 3, 1)).contiguous()
            weight = torch.permute(weight, (0, 2, 3, 1)).contiguous()

        convs = [
            (
                torch.ops.cadence.quantized_conv_nchw.per_tensor
                if memory_format == torch.contiguous_format
                else torch.ops.cadence.quantized_conv_nhwc.per_tensor
            )
        ]

        optimized_convs = []
        if input_tensor.dtype == torch.int8 and weight.dtype == torch.int8:
            if memory_format == torch.contiguous_format:
                optimized_convs = [
                    torch.ops.cadence.quantized_conv_nchw_asym8sxsym8s_asym8s.per_tensor,
                    torch.ops.cadence.quantized_conv_nchw_dilated_asym8sxsym8s_asym8s.per_tensor,
                    torch.ops.cadence.quantized_conv_nchw_depthwise_asym8sxsym8s_asym8s.per_tensor,
                ]

            else:
                optimized_convs = [
                    torch.ops.cadence.quantized_conv_nhwc_asym8sxsym8s_asym8s.per_tensor,
                    torch.ops.cadence.quantized_conv_nhwc_dilated_asym8sxsym8s_asym8s.per_tensor,
                    torch.ops.cadence.quantized_conv_nhwc_depthwise_asym8sxsym8s_asym8s.per_tensor,
                ]
        elif input_tensor.dtype == torch.uint8 and weight.dtype == torch.uint8:
            if memory_format == torch.contiguous_format:
                optimized_convs = [
                    torch.ops.cadence.quantized_conv_nchw_asym8uxsym8u_asym8u.per_tensor,
                    torch.ops.cadence.quantized_conv_nchw_dilated_asym8uxsym8u_asym8u.per_tensor,
                    torch.ops.cadence.quantized_conv_nchw_depthwise_asym8uxsym8u_asym8u.per_tensor,
                ]

            else:
                optimized_convs = [
                    torch.ops.cadence.quantized_conv_nhwc_asym8uxsym8u_asym8u.per_tensor,
                    torch.ops.cadence.quantized_conv_nhwc_dilated_asym8uxsym8u_asym8u.per_tensor,
                    torch.ops.cadence.quantized_conv_nhwc_depthwise_asym8uxsym8u_asym8u.per_tensor,
                ]

        convs.extend(optimized_convs)
        for conv in convs:
            output = conv(
                input_tensor,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                in_zero_point,
                weight_zero_point,
                bias_scale,
                output_scale,
                output_zero_point,
                out_multiplier,
                out_shift,
            ).to(memory_format=torch.contiguous_format)

            # Verify output properties
            self.assertEqual(output.dtype, dtype, f"Output dtype should be {dtype}")
            self.assertEqual(
                output.shape,
                expected_output.shape,
                "Output shape should match expected shape",
            )

            # Verify output matches expected values
            self.assertTrue(
                torch.equal(output, expected_output),
                f"Output values don't match expected. Got {output}, expected {expected_output}",
            )

    @expand(
        [
            # Test case 1: Basic int8 case with negative scale
            *[
                (
                    "basic_int8",
                    torch.tensor([-1, 0, 1, 3], dtype=dtype),  # input
                    0,  # X_zero_point (scalar broadcast)
                    0,  # out_zero_point
                    1073741824,  # out_multiplier (0.5 * 2^31)
                    0,  # out_shift
                    dtype,  # dtype
                    torch.tensor(
                        [0, 0, 0, -2], dtype=dtype
                    ),  # expected: relu(-1,0,1,3) = (0,0,1,3) * (-0.5) + 0 = (0,0,-0.5,-1.5) -> (0,0,0,-2)
                )
                for dtype in [torch.int8]
            ],
            # Test case 2: uint8 with non-zero zero point
            *[
                (
                    "uint8_with_zp",
                    torch.tensor([126, 128, 130, 132], dtype=dtype),  # input
                    128,  # X_zero_point
                    64,  # out_zero_point
                    536870912,  # out_multiplier (0.25 * 2^31)
                    0,  # out_shift
                    dtype,  # dtype
                    torch.tensor(
                        [64, 64, 64, 63], dtype=dtype
                    ),  # expected: relu(-2,0,2,4) = (0,0,2,4) * (-0.25) + 64 = (64,64,63.5,63) -> (64,64,64,63)
                )
                for dtype in [torch.uint8]
            ],
            # Test case 3: All negative values (should all become zero after ReLU)
            *[
                (
                    "all_negative_int8",
                    torch.tensor([-5, -3, -1], dtype=dtype),  # input
                    0,  # X_zero_point
                    10,  # out_zero_point
                    1073741824,  # out_multiplier (0.5 * 2^31)
                    0,  # out_shift
                    dtype,  # dtype
                    torch.tensor(
                        [10, 10, 10], dtype=dtype
                    ),  # expected: relu(-5,-3,-1) = (0,0,0) * (-0.5) + 10 = (10,10,10)
                )
                for dtype in [torch.int8]
            ],
            # Test case 4: All positive values with shift (scale becomes -0.25)
            *[
                (
                    "positive_with_shift",
                    torch.tensor([2, 4, 6, 8], dtype=dtype),  # input
                    1,  # X_zero_point
                    5,  # out_zero_point
                    1073741824,  # out_multiplier (0.5 * 2^31)
                    1,  # out_shift (multiply by 2^1 = 2)
                    dtype,  # dtype
                    torch.tensor(
                        [4, 2, 0, -2], dtype=dtype
                    ),  # expected: relu(1,3,5,7) = (1,3,5,7) * (-1.0) + 5 = (4,2,0,-2)
                )
                for dtype in [torch.int8, torch.uint8]
            ],
            # Test case 4: Non-per-tensor
            *[
                (
                    "non_per_tensor",
                    torch.tensor([-1, -2, -3, 1, 2, 3], dtype=dtype),  # input
                    torch.tensor([0, 0, 0, 1, 1, 1]),  # X_zero_point
                    5,  # out_zero_point
                    torch.tensor([1073741824]),  # out_multiplier (0.5 * 2^31)
                    torch.tensor([1]),  # out_shift (multiply by 2^1 = 2)
                    dtype,  # dtype
                    torch.tensor([5, 5, 5, 5, 4, 3], dtype=dtype),
                )
                for dtype in [torch.int8]
            ],
        ]
    )
    def test_quantized_relu(
        self,
        name: str,
        X: torch.Tensor,
        X_zero_point: torch.Tensor | int,
        out_zero_point: int,
        out_multiplier: torch.Tensor | int,
        out_shift: torch.Tensor | int,
        dtype: torch.dtype,
        expected_output: torch.Tensor,
    ) -> None:

        if isinstance(X_zero_point, int):
            assert isinstance(out_multiplier, int)
            assert isinstance(out_shift, int)

            match dtype:
                case torch.int8:
                    quantized_relu = (
                        torch.ops.cadence.quantized_relu_asym8s_asym8s.per_tensor
                    )
                case torch.uint8:
                    quantized_relu = (
                        torch.ops.cadence.quantized_relu_asym8u_asym8u.per_tensor
                    )
                case _:
                    quantized_relu = torch.ops.cadence.quantized_relu_per_tensor

            output = quantized_relu(
                X,
                X_zero_point,
                out_zero_point,
                out_multiplier,
                out_shift,
            )
        else:
            output = torch.ops.cadence.quantized_relu(
                X, X_zero_point, out_zero_point, out_multiplier, out_shift
            )

        # Verify output properties
        self.assertEqual(output.dtype, dtype, f"Output dtype should be {dtype}")
        self.assertEqual(output.shape, X.shape, "Output shape should match input shape")

        # Verify output matches expected values
        self.assertTrue(
            torch.equal(output, expected_output),
            f"Output values don't match expected in {name}. Got {output}, expected {expected_output}",
        )
