# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from contextlib import nullcontext

import torch
from executorch.backends.fused_quant.ops import (
    _bmm_meta,
    _embedding_meta,
    _permute_contiguous,
    _requantize_meta,
    QuantParamsStruct,
)
from torch._subclasses.fake_tensor import FakeTensorMode


def _create_qparams(
    dtype: torch.dtype = torch.int8,
) -> QuantParamsStruct[torch.Tensor]:
    return QuantParamsStruct(
        scale=torch.tensor([1.0]),
        zero_point=torch.tensor([0], dtype=torch.int64),
        dtype=dtype,
        quant_min=torch.iinfo(dtype).min,
        quant_max=torch.iinfo(dtype).max,
    )


def _create_per_channel_qparams(
    num_channels: int, dtype: torch.dtype = torch.int8
) -> QuantParamsStruct[torch.Tensor]:
    # A 1D scale has a single non-unary dim, so it classifies as per-channel and
    # its channel axis derives to 0.
    return QuantParamsStruct(
        scale=torch.ones(num_channels),
        zero_point=torch.zeros(num_channels, dtype=torch.int64),
        dtype=dtype,
        quant_min=torch.iinfo(dtype).min,
        quant_max=torch.iinfo(dtype).max,
    )


def _create_per_group_qparams(
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    dtype: torch.dtype = torch.int8,
    quant_min: int = -8,
    quant_max: int = 7,
) -> QuantParamsStruct[torch.Tensor]:
    """Per-group qparams. ``scale`` is a full-rank [num_channels, num_groups] tensor.

    Defaults model the W4 case (4-bit range stored in an int8 container).
    """
    if zero_point is None:
        zero_point = torch.zeros_like(scale, dtype=torch.int64)
    return QuantParamsStruct(
        scale=scale,
        zero_point=zero_point,
        dtype=dtype,
        quant_min=quant_min,
        quant_max=quant_max,
    )


def _ref_quantize_per_group(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Independent, loop-based per-group quantize reference (channel axis 0)."""
    num_channels, grouped_dim = tensor.shape
    num_groups = scale.shape[1]
    group_size = grouped_dim // num_groups
    out = torch.empty_like(tensor)
    for k in range(num_channels):
        for g in range(num_groups):
            cols = slice(g * group_size, (g + 1) * group_size)
            out[k, cols] = torch.clamp(
                torch.round(tensor[k, cols] / scale[k, g]) + zero_point[k, g],
                quant_min,
                quant_max,
            )
    return out.to(dtype)


def _ref_dequantize_per_group(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Independent, loop-based per-group dequantize reference (channel axis 0)."""
    num_channels, grouped_dim = tensor.shape
    num_groups = scale.shape[1]
    group_size = grouped_dim // num_groups
    out = torch.empty(tensor.shape, dtype=torch.float32)
    for k in range(num_channels):
        for g in range(num_groups):
            cols = slice(g * group_size, (g + 1) * group_size)
            out[k, cols] = (
                tensor[k, cols].to(torch.float32) - zero_point[k, g]
            ) * scale[k, g]
    return out


def _flat_qparams(
    qp: QuantParamsStruct[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype, int, int]:
    return (qp.scale, qp.zero_point, qp.dtype, qp.quant_min, qp.quant_max)


class TestQuantParamsFunctions(unittest.TestCase):
    def test_is_per_tensor(self) -> None:
        qparams = _create_qparams()
        self.assertTrue(qparams.is_per_tensor())
        self.assertFalse(qparams.is_per_channel())
        self.assertFalse(qparams.is_per_group())

    def test_is_per_channel(self) -> None:
        qparams = _create_per_channel_qparams(16)
        self.assertFalse(qparams.is_per_tensor())
        self.assertTrue(qparams.is_per_channel())
        self.assertFalse(qparams.is_per_group())

    def test_channel_axis_derivation(self) -> None:
        # all-ones -> 0; single non-unary dim -> that dim; multiple -> None.
        per_tensor = QuantParamsStruct(
            torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int64), torch.int8, -8, 7
        )
        self.assertEqual(per_tensor.channel_axis(), 0)
        per_channel_axis1 = QuantParamsStruct(
            torch.ones(1, 4), torch.zeros(1, 4, dtype=torch.int64), torch.int8, -8, 7
        )
        self.assertEqual(per_channel_axis1.channel_axis(), 1)
        blockwise = QuantParamsStruct(
            torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int64), torch.int8, -8, 7
        )
        self.assertIsNone(blockwise.channel_axis())

    def test_quant_min_max_in_qparams(self) -> None:
        qparams_int8 = _create_qparams(torch.int8)
        self.assertEqual(qparams_int8.quant_min, -128)
        self.assertEqual(qparams_int8.quant_max, 127)

        qparams_uint8 = _create_qparams(torch.uint8)
        self.assertEqual(qparams_uint8.quant_min, 0)
        self.assertEqual(qparams_uint8.quant_max, 255)

    def test_validate_succeeds(self) -> None:
        _create_qparams().validate()
        _create_per_channel_qparams(16).validate()
        _create_per_group_qparams(torch.ones(4, 4)).validate()

    def test_validate_shape_mismatch_raises(self) -> None:
        qparams = QuantParamsStruct(
            scale=torch.ones(4, 1),
            zero_point=torch.zeros(4, 2, dtype=torch.int64),
            dtype=torch.int8,
            quant_min=-8,
            quant_max=7,
        )
        with self.assertRaisesRegex(ValueError, "same shape"):
            qparams.validate()


class TestAffineQuant(unittest.TestCase):
    def test_per_tensor_round_trip(self) -> None:
        torch.manual_seed(0)
        scale = torch.tensor([0.5])
        zero_point = torch.tensor([0], dtype=torch.int64)
        quant_qp = QuantParamsStruct(scale, zero_point, torch.int8, -8, 7)
        dequant_qp = QuantParamsStruct(scale, zero_point, torch.float32, -8, 7)
        q = torch.randint(-8, 8, (3, 5), dtype=torch.int8)
        weight = dequant_qp.dequantize(q)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertTrue(torch.equal(quant_qp.quantize(weight), q))

    def test_per_channel_round_trip(self) -> None:
        torch.manual_seed(1)
        num_channels, in_features = 4, 6
        scale = torch.rand(num_channels, 1) + 0.1  # full-rank [M, 1] -> per-channel
        zero_point = torch.zeros(num_channels, 1, dtype=torch.int64)
        quant_qp = QuantParamsStruct(scale, zero_point, torch.int8, -8, 7)
        dequant_qp = QuantParamsStruct(scale, zero_point, torch.float32, -8, 7)
        q = torch.randint(-8, 8, (num_channels, in_features), dtype=torch.int8)
        weight = dequant_qp.dequantize(q)
        self.assertTrue(torch.equal(quant_qp.quantize(weight), q))


class TestRequantize(unittest.TestCase):
    def test_requantize(self) -> None:
        inp_qparams = QuantParamsStruct(
            torch.tensor([0.5]),
            torch.tensor([1], dtype=torch.int64),
            torch.float32,
            -128,
            127,
        )
        out_qparams = QuantParamsStruct(
            torch.tensor([0.25]),
            torch.tensor([3], dtype=torch.int64),
            torch.uint8,
            0,
            255,
        )
        inp = torch.tensor([[-4, 0, 7]], dtype=torch.int8)

        expected = out_qparams.quantize(inp_qparams.dequantize(inp))
        actual = torch.ops.fused_quant.requantize(
            inp, *_flat_qparams(inp_qparams), *_flat_qparams(out_qparams)
        )
        meta = _requantize_meta(
            inp, *_flat_qparams(inp_qparams), *_flat_qparams(out_qparams)
        )

        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(meta.dtype, torch.uint8)
        self.assertTrue(torch.equal(actual, expected))


class TestConvNd(unittest.TestCase):
    def test_rejects_wrong_spatial_dimensions(self) -> None:
        no_qparams = (None, None, torch.float32, 0, 0)
        for name, target, spatial_dims in (
            ("conv1d", torch.ops.fused_quant.conv1d.default, 1),
            ("conv2d", torch.ops.fused_quant.conv2d.default, 2),
            ("conv3d", torch.ops.fused_quant.conv3d.default, 3),
        ):
            wrong_spatial_dims = 2 if spatial_dims == 1 else 1
            for fake in (False, True):
                with (
                    self.subTest(name=name, fake=fake),
                    FakeTensorMode() if fake else nullcontext(),
                ):
                    inp = torch.randn(1, 2, *([8] * wrong_spatial_dims))
                    weight = torch.randn(3, 2, *([3] * wrong_spatial_dims))
                    with self.assertRaisesRegex(
                        ValueError,
                        f"fused_quant::{name} expects input and weight rank "
                        f"{spatial_dims + 2}",
                    ):
                        target(
                            inp,
                            weight,
                            None,
                            *no_qparams,
                            *no_qparams,
                            *no_qparams,
                            *no_qparams,
                            [1] * wrong_spatial_dims,
                            [0] * wrong_spatial_dims,
                            [1] * wrong_spatial_dims,
                            1,
                        )

    def test_rejects_wrong_spatial_argument_lengths(self) -> None:
        no_qparams = (None, None, torch.float32, 0, 0)
        inp = torch.randn(1, 2, 8, 8)
        weight = torch.randn(3, 2, 3, 3)
        for arg_name, stride, padding, dilation in (
            ("stride", [1], [0, 0], [1, 1]),
            ("padding", [1, 1], [0], [1, 1]),
            ("dilation", [1, 1], [0, 0], [1]),
        ):
            with (
                self.subTest(arg_name=arg_name),
                self.assertRaisesRegex(
                    ValueError,
                    f"fused_quant::conv2d expects {arg_name} to contain 2 values",
                ),
            ):
                torch.ops.fused_quant.conv2d.default(
                    inp,
                    weight,
                    None,
                    *no_qparams,
                    *no_qparams,
                    *no_qparams,
                    *no_qparams,
                    stride,
                    padding,
                    dilation,
                    1,
                )


class TestConvolutionChannelsLast(unittest.TestCase):
    def test_qparams_use_channels_last_layout(self) -> None:
        torch.manual_seed(0)
        inp = torch.randint(-8, 8, (1, 4, 4, 2), dtype=torch.int8)
        weight = torch.randint(-8, 8, (2, 3, 3, 2), dtype=torch.int8)
        inp_qparams = QuantParamsStruct(
            torch.tensor([[[[0.25, 0.5]]]]),
            torch.zeros(1, 1, 1, 2, dtype=torch.int64),
            torch.float32,
            -8,
            7,
        )
        weight_qparams = QuantParamsStruct(
            torch.tensor(
                [
                    [[[0.25, 0.5]]],
                    [[[0.75, 1.0]]],
                ]
            ),
            torch.zeros(2, 1, 1, 2, dtype=torch.int64),
            torch.float32,
            -8,
            7,
        )
        out_qparams = QuantParamsStruct(
            torch.tensor([[[[0.5, 0.25]]]]),
            torch.zeros(1, 1, 1, 2, dtype=torch.int64),
            torch.int8,
            -128,
            127,
        )

        dq_inp = inp_qparams.dequantize(inp).permute(0, 3, 1, 2)
        dq_weight = weight_qparams.dequantize(weight).permute(0, 3, 1, 2)
        expected = out_qparams.quantize(
            torch.nn.functional.conv2d(dq_inp, dq_weight).permute(0, 2, 3, 1)
        )
        actual = torch.ops.fused_quant.convolution_channels_last.default(
            inp,
            weight,
            None,
            *_flat_qparams(inp_qparams),
            *_flat_qparams(weight_qparams),
            None,
            None,
            torch.uint8,
            0,
            0,
            *_flat_qparams(out_qparams),
            [1, 1],
            [0, 0],
            [1, 1],
            False,
            [0, 0],
            1,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_transposed_convolution(self) -> None:
        inp = torch.randn(1, 4, 4, 2)
        weight = torch.randn(2, 3, 3, 3)
        no_qparams = (None, None, torch.float32, 0, 0)
        expected = torch.ops.aten.convolution.default(
            _permute_contiguous(inp, [0, 3, 1, 2]),
            _permute_contiguous(weight, [0, 3, 1, 2]),
            None,
            [2, 2],
            [1, 1],
            [1, 1],
            True,
            [1, 1],
            1,
        ).permute(0, 2, 3, 1)

        actual = torch.ops.fused_quant.convolution_channels_last.default(
            inp,
            weight,
            None,
            *no_qparams,
            *no_qparams,
            *no_qparams,
            *no_qparams,
            [2, 2],
            [1, 1],
            [1, 1],
            True,
            [1, 1],
            1,
        )

        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


class TestPermuteContiguous(unittest.TestCase):
    def test_canonicalizes_singleton_dimension_strides(self) -> None:
        tensor = torch.empty((16, 1, 1, 64))
        permuted = tensor.permute(0, 3, 1, 2)

        self.assertTrue(permuted.is_contiguous())
        self.assertEqual(permuted.stride(), (64, 1, 64, 64))
        self.assertEqual(
            _permute_contiguous(tensor, [0, 3, 1, 2]).stride(),
            (64, 1, 1, 1),
        )


class PoolingMetaTest(unittest.TestCase):
    def test_integer_inputs(self) -> None:
        inp_qparams = QuantParamsStruct(
            torch.tensor([1.0]),
            torch.tensor([0], dtype=torch.int64),
            torch.float32,
            -128,
            127,
        )
        out_qparams = _create_qparams(torch.uint8)

        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            inp_nchw = mode.from_tensor(torch.empty((1, 2, 4, 4), dtype=torch.int8))
            max_values, max_indices = (
                torch.ops.fused_quant.max_pool2d_with_indices.default(
                    inp_nchw,
                    *_flat_qparams(inp_qparams),
                    *_flat_qparams(out_qparams),
                    [2, 2],
                    [2, 2],
                    [0, 0],
                    [1, 1],
                    False,
                )
            )
            avg_nchw = torch.ops.fused_quant.avg_pool2d.default(
                inp_nchw,
                *_flat_qparams(inp_qparams),
                *_flat_qparams(out_qparams),
                [2, 2],
                [2, 2],
                [0, 0],
                False,
                True,
                None,
            )

            inp_nhwc = mode.from_tensor(torch.empty((1, 4, 4, 2), dtype=torch.int8))
            avg_nhwc = torch.ops.fused_quant.avg_pool2d_channels_last.default(
                inp_nhwc,
                *_flat_qparams(inp_qparams),
                *_flat_qparams(out_qparams),
                [2, 2],
                [2, 2],
                [0, 0],
                False,
                True,
                None,
            )

        self.assertEqual(max_values.shape, torch.Size([1, 2, 2, 2]))
        self.assertEqual(max_values.dtype, torch.uint8)
        self.assertEqual(max_indices.shape, torch.Size([1, 2, 2, 2]))
        self.assertEqual(max_indices.dtype, torch.int64)
        self.assertEqual(avg_nchw.shape, torch.Size([1, 2, 2, 2]))
        self.assertEqual(avg_nchw.dtype, torch.uint8)
        self.assertEqual(avg_nhwc.shape, torch.Size([1, 2, 2, 2]))
        self.assertEqual(avg_nhwc.dtype, torch.uint8)


class TestMaxPool2dWithIndicesChannelsLast(unittest.TestCase):
    def test_qparams_use_channels_last_layout(self) -> None:
        inp = torch.randint(-8, 8, (1, 4, 4, 2), dtype=torch.int8)
        inp_qparams = QuantParamsStruct(
            torch.tensor([[[[0.25, 0.5]]]]),
            torch.zeros(1, 1, 1, 2, dtype=torch.int64),
            torch.float32,
            -8,
            7,
        )
        out_qparams = QuantParamsStruct(
            torch.tensor([[[[0.5, 0.25]]]]),
            torch.zeros(1, 1, 1, 2, dtype=torch.int64),
            torch.int8,
            -128,
            127,
        )

        values_nchw, indices_nchw = torch.ops.aten.max_pool2d_with_indices.default(
            inp_qparams.dequantize(inp).permute(0, 3, 1, 2),
            [2, 2],
            [2, 2],
        )
        expected_values = out_qparams.quantize(values_nchw.permute(0, 2, 3, 1))
        expected_indices = indices_nchw.permute(0, 2, 3, 1)
        actual_values, actual_indices = (
            torch.ops.fused_quant.max_pool2d_with_indices_channels_last.default(
                inp,
                *_flat_qparams(inp_qparams),
                *_flat_qparams(out_qparams),
                [2, 2],
                [2, 2],
                [0, 0],
                [1, 1],
                False,
            )
        )
        self.assertTrue(torch.equal(actual_values, expected_values))
        self.assertTrue(torch.equal(actual_indices, expected_indices))


class TestAvgPool2dChannelsLast(unittest.TestCase):
    def test_output_has_canonical_strides_with_singleton_channel(self) -> None:
        inp = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
        no_qparams = (None, None, torch.float32, -2147483648, 2147483647)

        actual = torch.ops.fused_quant.avg_pool2d_channels_last.default(
            inp,
            *no_qparams,
            *no_qparams,
            [2, 2],
            [2, 2],
            [0, 0],
            False,
            False,
            None,
        )
        expected = torch.ops.aten.avg_pool2d.default(
            inp.permute(0, 3, 1, 2),
            [2, 2],
            [2, 2],
        ).permute(0, 2, 3, 1)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.stride(), (4, 2, 1, 1))

    def test_qparams_and_aten_arguments(self) -> None:
        inp = torch.randint(-8, 8, (1, 4, 4, 2), dtype=torch.int8)
        inp_qparams = QuantParamsStruct(
            torch.tensor([[[[0.25, 0.5]]]]),
            torch.zeros(1, 1, 1, 2, dtype=torch.int64),
            torch.float32,
            -8,
            7,
        )
        out_qparams = QuantParamsStruct(
            torch.tensor([[[[0.5, 0.25]]]]),
            torch.zeros(1, 1, 1, 2, dtype=torch.int64),
            torch.int8,
            -128,
            127,
        )

        for ceil_mode, count_include_pad, divisor_override in (
            (False, False, None),
            (True, True, 5),
        ):
            with self.subTest(
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            ):
                output_nchw = torch.ops.aten.avg_pool2d.default(
                    inp_qparams.dequantize(inp).permute(0, 3, 1, 2),
                    [2, 2],
                    [2, 2],
                    [1, 1],
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )
                expected = out_qparams.quantize(output_nchw.permute(0, 2, 3, 1))
                actual = torch.ops.fused_quant.avg_pool2d_channels_last.default(
                    inp,
                    *_flat_qparams(inp_qparams),
                    *_flat_qparams(out_qparams),
                    [2, 2],
                    [2, 2],
                    [1, 1],
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )
                self.assertTrue(torch.equal(actual, expected))


class TestPerGroupQuant(unittest.TestCase):
    def test_is_per_group_classification(self) -> None:
        qparams = _create_per_group_qparams(torch.ones(4, 4))
        self.assertTrue(qparams.is_per_group())
        self.assertFalse(qparams.is_per_tensor())
        self.assertFalse(qparams.is_per_channel())

    def test_quantize_matches_loop_reference(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(4, 16)
        scale = torch.rand(4, 4) + 0.1  # strictly positive, num_groups=4
        qparams = _create_per_group_qparams(scale)
        expected = _ref_quantize_per_group(
            weight, scale, qparams.zero_point, -8, 7, torch.int8
        )
        actual = qparams.quantize(weight)
        self.assertEqual(actual.dtype, torch.int8)
        self.assertTrue(torch.equal(actual, expected))

    def test_dequantize_matches_loop_reference(self) -> None:
        torch.manual_seed(1)
        scale = torch.rand(4, 4) + 0.1
        qweight = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
        # A dequant qparams carries the float out_dtype.
        qparams = _create_per_group_qparams(scale, dtype=torch.float32)
        expected = _ref_dequantize_per_group(qweight, scale, qparams.zero_point)
        actual = qparams.dequantize(qweight)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, expected))

    def test_different_groups_use_different_scales(self) -> None:
        # Two groups along the contraction axis with very different scales.
        scale = torch.tensor([[1.0, 10.0]])  # 1 channel, 2 groups
        qparams = _create_per_group_qparams(scale)
        weight = torch.tensor([[4.0, 4.0, 40.0, 40.0]])  # group0 / 1, group1 / 10
        actual = qparams.quantize(weight)
        # group0: round(4/1)=4 ; group1: round(40/10)=4 -> all 4s despite 10x range
        self.assertTrue(
            torch.equal(actual, torch.tensor([[4, 4, 4, 4]], dtype=torch.int8))
        )

    def test_round_trip_on_grid_is_exact(self) -> None:
        torch.manual_seed(2)
        scale = torch.rand(4, 4) + 0.1
        zero_point = torch.zeros(4, 4, dtype=torch.int64)
        quant_qp = _create_per_group_qparams(
            scale, zero_point=zero_point, dtype=torch.int8
        )
        dequant_qp = _create_per_group_qparams(
            scale, zero_point=zero_point, dtype=torch.float32
        )
        q = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
        # Build a float weight that lies exactly on the per-group grid, then
        # re-quantizing must recover the original integer codes.
        weight = dequant_qp.dequantize(q)
        requantized = quant_qp.quantize(weight)
        self.assertTrue(torch.equal(requantized, q))

    def test_quantize_clamps_out_of_range(self) -> None:
        scale = torch.tensor([[1.0, 1.0]])
        qparams = _create_per_group_qparams(scale)
        weight = torch.tensor([[100.0, -100.0, 3.0, -3.0]])
        actual = qparams.quantize(weight)
        self.assertTrue(
            torch.equal(actual, torch.tensor([[7, -8, 3, -3]], dtype=torch.int8))
        )

    def test_indivisible_grouped_dim_raises(self) -> None:
        scale = torch.ones(4, 3)  # 3 groups
        qparams = _create_per_group_qparams(scale)
        weight = torch.randn(4, 16)  # 16 not divisible by 3
        with self.assertRaisesRegex(ValueError, "must be divisible by"):
            qparams.quantize(weight)

    def test_rank_mismatch_raises(self) -> None:
        # A non-singleton scale must be full-rank (match the tensor rank).
        scale = torch.ones(4, 4)
        qparams = _create_per_group_qparams(scale)
        weight = torch.randn(4, 4, 4)
        with self.assertRaisesRegex(ValueError, "match the tensor rank"):
            qparams.quantize(weight)

    def test_nonzero_zero_point(self) -> None:
        scale = torch.tensor([[2.0, 2.0]])
        zero_point = torch.tensor([[1, -1]], dtype=torch.int64)
        quant_qp = _create_per_group_qparams(
            scale, zero_point=zero_point, dtype=torch.int8
        )
        weight = torch.tensor([[4.0, 4.0, 4.0, 4.0]])
        # group0: round(4/2)+1=3 ; group1: round(4/2)-1=1
        actual = quant_qp.quantize(weight)
        self.assertTrue(
            torch.equal(actual, torch.tensor([[3, 3, 1, 1]], dtype=torch.int8))
        )
        # Dequantize inverts: group0 (3-1)*2=4 ; group1 (1+1)*2=4
        dequant_qp = _create_per_group_qparams(
            scale, zero_point=zero_point, dtype=torch.float32
        )
        dq = dequant_qp.dequantize(actual)
        self.assertTrue(torch.allclose(dq, torch.tensor([[4.0, 4.0, 4.0, 4.0]])))


def _embedding_weight_qp(
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    quant_min: int = -8,
    quant_max: int = 7,
) -> QuantParamsStruct[torch.Tensor]:
    """The table (dequantize) qparams for fused_quant.embedding.

    dtype is float32 because the qparams describes dequantizing the quantized
    table to float embeddings. Granularity is encoded by the scale shape relative
    to the [num_embeddings, embedding_dim] table: a [num_embeddings, 1] scale is
    per-row (per-channel), and a [num_embeddings, num_groups] scale is per-group
    over the embedding dimension.
    """
    if zero_point is None:
        zero_point = torch.zeros_like(scale, dtype=torch.int64)
    return QuantParamsStruct(
        scale=scale,
        zero_point=zero_point,
        dtype=torch.float32,
        quant_min=quant_min,
        quant_max=quant_max,
    )


# A null output-qparams block: indices.shape + [emb_dim] embeddings stay float.
_NULL_OUT_QP_ARGS: tuple[None, None, torch.dtype, int, int] = (
    None,
    None,
    torch.float32,
    0,
    0,
)


def _call_embedding(
    qtable: torch.Tensor,
    indices: torch.Tensor,
    weight_qp: QuantParamsStruct[torch.Tensor],
    out_qp: QuantParamsStruct[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Invoke fused_quant.embedding: (table, weight_qp block, out_qp block, indices)."""
    out_flat = _flat_qparams(out_qp) if out_qp is not None else _NULL_OUT_QP_ARGS
    return torch.ops.fused_quant.embedding(
        qtable, *_flat_qparams(weight_qp), *out_flat, indices
    )


class TestEmbedding(unittest.TestCase):
    def test_embedding_per_channel(self) -> None:
        torch.manual_seed(0)
        num_embeddings, embedding_dim = 6, 8
        qtable = torch.randint(-8, 8, (num_embeddings, embedding_dim), dtype=torch.int8)
        scale = torch.rand(num_embeddings, 1) + 0.1  # per-row
        weight_qp = _embedding_weight_qp(scale)
        indices = torch.tensor([0, 3, 5, 1])
        expected = weight_qp.dequantize(qtable)[indices]
        actual = _call_embedding(qtable, indices, weight_qp)
        self.assertEqual(tuple(actual.shape), (4, embedding_dim))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual, expected))

    def test_embedding_per_group(self) -> None:
        torch.manual_seed(1)
        num_embeddings, embedding_dim, num_groups = 6, 8, 4
        qtable = torch.randint(-8, 8, (num_embeddings, embedding_dim), dtype=torch.int8)
        scale = torch.rand(num_embeddings, num_groups) + 0.1  # per-group over emb dim
        weight_qp = _embedding_weight_qp(scale)
        indices = torch.tensor([[0, 1], [5, 2]])  # 2D indices
        expected = weight_qp.dequantize(qtable)[indices]
        actual = _call_embedding(qtable, indices, weight_qp)
        self.assertEqual(tuple(actual.shape), (2, 2, embedding_dim))
        self.assertTrue(torch.allclose(actual, expected))

    def test_embedding_quantized_output(self) -> None:
        # Exercises the optional output qparams block: the gathered embeddings are
        # requantized to int8 (per-tensor).
        torch.manual_seed(2)
        num_embeddings, embedding_dim = 6, 8
        qtable = torch.randint(-8, 8, (num_embeddings, embedding_dim), dtype=torch.int8)
        scale = torch.rand(num_embeddings, 1) + 0.1
        weight_qp = _embedding_weight_qp(scale)
        out_qp = QuantParamsStruct(
            scale=torch.tensor([0.05]),
            zero_point=torch.tensor([0], dtype=torch.int64),
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        )
        indices = torch.tensor([0, 2, 4])
        expected = out_qp.quantize(weight_qp.dequantize(qtable)[indices])
        actual = _call_embedding(qtable, indices, weight_qp, out_qp)
        self.assertEqual(actual.dtype, torch.int8)
        self.assertTrue(torch.equal(actual, expected))

    def test_embedding_requires_quantized_table(self) -> None:
        qtable = torch.randint(-8, 8, (6, 8), dtype=torch.int8)
        indices = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "At least one"):
            torch.ops.fused_quant.embedding(
                qtable, *_NULL_OUT_QP_ARGS, *_NULL_OUT_QP_ARGS, indices
            )

    def test_embedding_non_2d_table_raises(self) -> None:
        qtable = torch.randint(-8, 8, (6, 8, 2), dtype=torch.int8)
        scale = torch.rand(6) + 0.1
        weight_qp = _embedding_weight_qp(scale)
        indices = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            _call_embedding(qtable, indices, weight_qp)

    def test_embedding_padding_idx_not_supported(self) -> None:
        qtable = torch.randint(-8, 8, (6, 8), dtype=torch.int8)
        weight_qp = _embedding_weight_qp(torch.rand(6, 1) + 0.1)
        indices = torch.tensor([0, 1])
        with self.assertRaises(AssertionError):
            torch.ops.fused_quant.embedding(
                qtable,
                *_flat_qparams(weight_qp),
                *_NULL_OUT_QP_ARGS,
                indices,
                0,  # padding_idx (not the aten default -1)
                False,  # scale_grad_by_freq
                False,  # sparse
            )

    def test_embedding_scale_grad_by_freq_not_supported(self) -> None:
        qtable = torch.randint(-8, 8, (6, 8), dtype=torch.int8)
        weight_qp = _embedding_weight_qp(torch.rand(6, 1) + 0.1)
        indices = torch.tensor([0, 1])
        with self.assertRaises(AssertionError):
            torch.ops.fused_quant.embedding(
                qtable,
                *_flat_qparams(weight_qp),
                *_NULL_OUT_QP_ARGS,
                indices,
                -1,  # padding_idx
                True,  # scale_grad_by_freq
                False,  # sparse
            )

    def test_embedding_sparse_not_supported(self) -> None:
        qtable = torch.randint(-8, 8, (6, 8), dtype=torch.int8)
        weight_qp = _embedding_weight_qp(torch.rand(6, 1) + 0.1)
        indices = torch.tensor([0, 1])
        with self.assertRaises(AssertionError):
            torch.ops.fused_quant.embedding(
                qtable,
                *_flat_qparams(weight_qp),
                *_NULL_OUT_QP_ARGS,
                indices,
                -1,  # padding_idx
                False,  # scale_grad_by_freq
                True,  # sparse
            )

    def test_embedding_meta_shape(self) -> None:
        scale = torch.rand(6, 1) + 0.1
        weight_qp = _embedding_weight_qp(scale)
        qtable = torch.empty(6, 8, dtype=torch.int8)
        indices = torch.zeros(3, 4, dtype=torch.int64)
        out = _embedding_meta(
            qtable, *_flat_qparams(weight_qp), *_NULL_OUT_QP_ARGS, indices
        )
        self.assertEqual(tuple(out.shape), (3, 4, 8))
        self.assertEqual(out.dtype, torch.float32)

    def test_embedding_meta_dtype_follows_table_dequant_dtype(self) -> None:
        # When the output is not quantized, the dtype is the table's dequantize
        # output dtype (here bfloat16), not assumed float32.
        weight_qp = QuantParamsStruct(
            scale=torch.rand(6, 1) + 0.1,
            zero_point=torch.zeros(6, 1, dtype=torch.int64),
            dtype=torch.bfloat16,
            quant_min=-8,
            quant_max=7,
        )
        qtable = torch.empty(6, 8, dtype=torch.int8)
        indices = torch.zeros(3, dtype=torch.int64)
        out = _embedding_meta(
            qtable, *_flat_qparams(weight_qp), *_NULL_OUT_QP_ARGS, indices
        )
        self.assertEqual(out.dtype, torch.bfloat16)


class TestOps(unittest.TestCase):
    def test_bmm_incorrect_dimension(self) -> None:
        qp = _flat_qparams(_create_qparams())
        with self.assertRaisesRegex(ValueError, "Input tensors must be 3D"):
            _bmm_meta(
                torch.randn(2, 3),
                torch.randn(2, 3, 5),
                *qp,
                *qp,
                *qp,
            )

    def test_bmm_incorrect_batch_dimension(self) -> None:
        qp = _flat_qparams(_create_qparams())
        with self.assertRaisesRegex(
            ValueError, "Input tensors must have the same batch dimension"
        ):
            _bmm_meta(
                torch.randn(2, 3, 5),
                torch.randn(3, 5, 3),
                *qp,
                *qp,
                *qp,
            )

    def test_bmm_incorrect_inner_dimension(self) -> None:
        qp = _flat_qparams(_create_qparams())
        with self.assertRaisesRegex(
            ValueError, "Input tensors must have the same inner dimension"
        ):
            _bmm_meta(
                torch.randn(2, 3, 5),
                torch.randn(2, 4, 5),
                *qp,
                *qp,
                *qp,
            )
