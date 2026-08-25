# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    required_cmsis_nn_buffer_sizes,
)
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def _run_conv2d(op, x, weight, bias):
    out_channels = weight.shape[0]
    return op(
        x,
        weight,
        bias,
        [1, 1],
        [1, 1],
        [1, 1],
        5,
        -3,
        torch.full((out_channels,), 1 << 30, dtype=torch.int32),
        torch.full((out_channels,), -2, dtype=torch.int32),
        -128,
        127,
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_depthwise_conv2d(op, x, weight, bias):
    out_channels = weight.shape[3]
    return op(
        x,
        weight,
        bias,
        [1, 1],
        [1, 1],
        [1, 1],
        1,
        5,
        -3,
        torch.full((out_channels,), 1 << 30, dtype=torch.int32),
        torch.full((out_channels,), -2, dtype=torch.int32),
        -128,
        127,
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_transpose_conv2d(op, x, weight, bias):
    out_channels = weight.shape[0]
    return op(
        x,
        weight,
        bias,
        [2, 2],
        [1, 1],
        [0, 0],
        [1, 1],
        5,
        -3,
        torch.full((out_channels,), 1 << 30, dtype=torch.int32),
        torch.full((out_channels,), -2, dtype=torch.int32),
        -128,
        127,
        torch.zeros(0, dtype=torch.uint8),
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_avg_pool2d(op, x):
    return op(
        x,
        [2, 2],
        [2, 2],
        [0, 0],
        False,
        0,
        1 << 30,
        1,
        torch.zeros(0, dtype=torch.uint8),
    )


def _run_max_pool2d(op, x):
    return op(
        x,
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        False,
        0,
        0,
        -128,
        127,
    )


def test_nhwc_conv2d_matches_legacy_layout():
    torch.manual_seed(0)
    x = torch.randint(-8, 8, (1, 3, 8, 8), dtype=torch.int8)
    weight = torch.randint(-4, 4, (4, 3, 3, 3), dtype=torch.int8)
    bias = torch.randint(-50, 50, (4,), dtype=torch.int32)

    legacy = _run_conv2d(
        torch.ops.cortex_m.quantized_conv2d,
        x.to(memory_format=torch.channels_last),
        weight,
        bias,
    )
    explicit = _run_conv2d(
        torch.ops.cortex_m.quantized_conv2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
        weight,
        bias,
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_depthwise_conv2d_matches_legacy_layout():
    torch.manual_seed(0)
    x = torch.randint(-8, 8, (1, 4, 8, 8), dtype=torch.int8)
    weight = torch.randint(-4, 4, (1, 3, 3, 4), dtype=torch.int8)
    bias = torch.randint(-50, 50, (4,), dtype=torch.int32)

    legacy = _run_depthwise_conv2d(
        torch.ops.cortex_m.quantized_depthwise_conv2d,
        x.to(memory_format=torch.channels_last),
        weight,
        bias,
    )
    explicit = _run_depthwise_conv2d(
        torch.ops.cortex_m.quantized_depthwise_conv2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
        weight,
        bias,
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_transpose_conv2d_matches_legacy_layout():
    torch.manual_seed(0)
    x = torch.randint(-8, 8, (1, 3, 6, 6), dtype=torch.int8)
    weight = torch.randint(-4, 4, (4, 3, 3, 3), dtype=torch.int8)
    bias = torch.randint(-50, 50, (4,), dtype=torch.int32)

    legacy = _run_transpose_conv2d(
        torch.ops.cortex_m.quantized_transpose_conv2d,
        x.to(memory_format=torch.channels_last),
        weight,
        bias,
    )
    explicit = _run_transpose_conv2d(
        torch.ops.cortex_m.quantized_transpose_conv2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
        weight,
        bias,
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_avg_pool2d_matches_legacy_layout():
    x = torch.randint(-8, 8, (1, 4, 8, 8), dtype=torch.int8)

    legacy = _run_avg_pool2d(
        torch.ops.cortex_m.quantized_avg_pool2d,
        x.to(memory_format=torch.channels_last),
    )
    explicit = _run_avg_pool2d(
        torch.ops.cortex_m.quantized_avg_pool2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_max_pool2d_matches_legacy_layout():
    x = torch.randint(-8, 8, (1, 4, 8, 8), dtype=torch.int8)

    legacy = _run_max_pool2d(
        torch.ops.cortex_m.quantized_max_pool2d,
        x.to(memory_format=torch.channels_last),
    )
    explicit = _run_max_pool2d(
        torch.ops.cortex_m.quantized_max_pool2d_nhwc,
        x.permute(0, 2, 3, 1).contiguous(),
    )

    torch.testing.assert_close(explicit, legacy.permute(0, 2, 3, 1))


def test_nhwc_conv2d_fake_shape_is_logical_nhwc():
    with FakeTensorMode():
        output = _run_conv2d(
            torch.ops.cortex_m.quantized_conv2d_nhwc,
            torch.empty(2, 10, 6, 3, dtype=torch.int8),
            torch.empty(5, 3, 3, 3, dtype=torch.int8),
            torch.empty(5, dtype=torch.int32),
        )

    assert output.shape == torch.Size([2, 10, 6, 5])
    assert output.dim_order() == (0, 1, 2, 3)


def test_pad_contiguous_preserves_singleton_height_layout():
    x = torch.arange(1 * 1 * 5 * 3, dtype=torch.int8).reshape(1, 1, 5, 3)
    pre_pad = [0, 0, 1, 0]
    post_pad = [0, 0, 2, 0]

    actual = torch.ops.cortex_m.pad_contiguous(x, pre_pad, post_pad, -7)
    expected = torch.nn.functional.pad(x, (0, 0, 1, 2, 0, 0, 0, 0), value=-7)

    assert actual.shape == torch.Size([1, 1, 8, 3])
    torch.testing.assert_close(actual, expected)


def test_pad_contiguous_handles_every_supported_rank():
    """Ranks below four are contiguous by construction, so this entry point
    covers them too. That is what lets it eventually replace ``cortex_m::pad``
    outright rather than sitting beside it forever."""
    for shape, pad in (
        ((2, 3, 4, 5), [0, 0, 1, 2]),
        ((2, 3, 4), [0, 0, 1, 2]),
        ((3, 5), [0, 0, 1, 2]),
        ((6,), [0, 0, 0, 2]),
    ):
        x = torch.randint(-8, 8, shape, dtype=torch.int8)
        rank = len(shape)
        offset = 4 - rank
        actual = torch.ops.cortex_m.pad_contiguous(x, pad, pad, -7)
        flat = []
        for dim in reversed(range(rank)):
            flat.extend([pad[offset + dim], pad[offset + dim]])
        expected = torch.nn.functional.pad(x, flat, value=-7)
        assert actual.shape == expected.shape, shape
        torch.testing.assert_close(actual, expected)


def test_pad_contiguous_rejects_invalid_padding_length():
    with pytest.raises(RuntimeError, match="expects four padding values per side"):
        torch.ops.cortex_m.pad_contiguous(
            torch.zeros((1, 1, 5, 3), dtype=torch.int8),
            [0, 0, 0],
            [0, 0, 0, 0],
            0,
        )

    with FakeTensorMode():
        with pytest.raises(RuntimeError, match="expects four padding values per side"):
            torch.ops.cortex_m.pad_contiguous(
                torch.zeros((1, 1, 5, 3), dtype=torch.int8),
                [0, 0, 0],
                [0, 0, 0, 0],
                0,
            )


def test_nhwc_and_legacy_scratch_sizes_match():
    backends = tuple(
        CortexMTargetConfig(cpu=cpu).backend for cpu in (CortexM.M33, CortexM.M55)
    )

    def make_node(
        target,
        input_shape,
        output_shape,
        weight_shape,
        trailing_args,
    ):
        graph = torch.fx.Graph()
        with FakeTensorMode() as mode:
            input_node = graph.placeholder("input")
            input_node.meta["val"] = mode.from_tensor(
                torch.empty(input_shape, dtype=torch.int8)
            )
            args = [input_node]
            if weight_shape is not None:
                weight_node = graph.placeholder("weight")
                weight_node.meta["val"] = mode.from_tensor(
                    torch.empty(weight_shape, dtype=torch.int8)
                )
                args.extend((weight_node, None))
            args.extend(trailing_args)
            node = graph.call_function(target, args=tuple(args))
            node.meta["val"] = mode.from_tensor(
                torch.empty(output_shape, dtype=torch.int8)
            )
        return node

    cases = (
        (
            exir_ops.edge.cortex_m.quantized_conv2d.default,
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
            (1, 3, 10, 6),
            (1, 10, 6, 3),
            (1, 4, 10, 6),
            (1, 10, 6, 4),
            (4, 3, 3, 3),
            ([1, 1], [1, 1], [1, 1], 5, -3, None, None, -128, 127, None),
        ),
        (
            exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default,
            exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default,
            (1, 4, 10, 6),
            (1, 10, 6, 4),
            (1, 4, 5, 5),
            (1, 5, 5, 4),
            (1, 3, 2, 4),
            ([2, 1], [1, 0], [1, 1], 1, 5, -3, None, None, -128, 127, None),
        ),
        (
            exir_ops.edge.cortex_m.quantized_transpose_conv2d.default,
            exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default,
            (1, 3, 5, 6),
            (1, 5, 6, 3),
            (1, 4, 9, 8),
            (1, 9, 8, 4),
            (4, 2, 3, 3),
            (
                [2, 1],
                [1, 0],
                [0, 0],
                [1, 1],
                5,
                -3,
                None,
                None,
                -128,
                127,
                None,
                None,
            ),
        ),
        (
            exir_ops.edge.cortex_m.quantized_avg_pool2d.default,
            exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default,
            (1, 4, 10, 6),
            (1, 10, 6, 4),
            (1, 4, 5, 3),
            (1, 5, 3, 4),
            None,
            ([2, 2], [2, 2], [0, 0], False, 0, 1 << 30, 1, None),
        ),
    )

    for (
        legacy_target,
        explicit_target,
        legacy_input_shape,
        explicit_input_shape,
        legacy_output_shape,
        explicit_output_shape,
        weight_shape,
        trailing_args,
    ) in cases:
        legacy = make_node(
            legacy_target,
            legacy_input_shape,
            legacy_output_shape,
            weight_shape,
            trailing_args,
        )
        explicit = make_node(
            explicit_target,
            explicit_input_shape,
            explicit_output_shape,
            weight_shape,
            trailing_args,
        )
        for backend in backends:
            assert required_cmsis_nn_buffer_sizes(
                legacy, backend
            ) == required_cmsis_nn_buffer_sizes(explicit, backend)
