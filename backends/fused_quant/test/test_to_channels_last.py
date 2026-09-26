# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import operator
import unittest
from typing import Any

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.optimization_passes.to_channels_last import (
    ToChannelsLast,
)
from executorch.backends.fused_quant.test.helpers import (
    create_per_axis_qparams,
    create_per_channel_qparams,
    create_per_tensor_qparams,
)
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir.dialects._ops import ops as exir_ops
from parameterized import parameterized
from torch.export import ExportedProgram

# These follow aten.convolution's argument order.
_CONVOLUTION1D_ARGS: tuple[Any, ...] = ([1], [0], [1], False, [0], 1)
_CONVOLUTION2D_ARGS: tuple[Any, ...] = (
    [1, 1],
    [0, 0],
    [1, 1],
    False,
    [0, 0],
    1,
)
_CONVOLUTION3D_ARGS: tuple[Any, ...] = (
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 1],
    False,
    [0, 0, 0],
    1,
)


def _build_fused_quant_conv_graph(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    bias_shape: tuple[int, ...] | None,
    conv_args: tuple[Any, ...],
    per_channel_activations: bool = False,
) -> tuple[ExportedProgram, tuple[torch.Tensor, ...]]:
    """Build a graph with a fused_quant.convolution edge op for testing.

    Quant params are lifted CONSTANT_TENSOR/BUFFER placeholders (matching how
    production lowers them), so they sit at the front of the graph rather than
    as inline body nodes.
    """
    inp = torch.randn(*input_shape).to(torch.int8)
    weight = torch.randn(*weight_shape).to(torch.int8)
    bias = torch.randn(*bias_shape) if bias_shape else None

    builder = ProgramBuilder()

    # Create placeholders first
    inp_placeholder = builder.placeholder("inp", inp)
    weight_placeholder = builder.placeholder("weight", weight)
    bias_placeholder = builder.placeholder("bias", bias) if bias is not None else None

    stride, padding, dilation, transposed, output_padding, groups = conv_args

    # Create qparams as lifted front placeholders
    inp_qparams = (
        create_per_axis_qparams(
            builder,
            input_shape[1],
            len(input_shape),
            axis=1,
            dtype=torch.float32,
            vary_values=True,
        )
        if per_channel_activations
        else create_per_tensor_qparams(builder, dtype=torch.float32)
    )
    weight_qparams = create_per_channel_qparams(
        builder,
        weight_shape[0],
        dtype=torch.float32,
        weight_ndim=len(weight_shape),
    )
    out_qparams = (
        create_per_axis_qparams(
            builder,
            weight_shape[0],
            len(input_shape),
            axis=1,
            vary_values=True,
        )
        if per_channel_activations
        else create_per_tensor_qparams(builder)
    )

    # Use edge op directly instead of ATen op
    conv_result = builder.call_operator(
        op=exir_ops.edge.fused_quant.convolution.default,
        args=(
            inp_placeholder,
            weight_placeholder,
            bias_placeholder,
            *inp_qparams,
            *weight_qparams,
            None,
            None,
            torch.uint8,
            0,
            0,
            *out_qparams,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ),
    )

    builder.output([conv_result])
    program = builder.get_program()

    if bias is not None:
        example_inputs = (inp, weight, bias)
    else:
        example_inputs = (inp, weight)

    return program, example_inputs


def _build_fused_quant_max_pool2d_with_indices_graph(
    input_shape: tuple[int, ...],
    pool_args: tuple[Any, ...],
    per_channel_qparams: bool = False,
) -> tuple[ExportedProgram, torch.Tensor]:
    inp = torch.randn(*input_shape).to(torch.int8)
    builder = ProgramBuilder()
    inp_placeholder = builder.placeholder("inp", inp)
    inp_qparams = (
        create_per_axis_qparams(
            builder,
            input_shape[1],
            len(input_shape),
            axis=1,
            dtype=torch.float32,
            vary_values=True,
        )
        if per_channel_qparams
        else create_per_tensor_qparams(builder, dtype=torch.float32)
    )
    out_qparams = (
        create_per_axis_qparams(
            builder,
            input_shape[1],
            len(input_shape),
            axis=1,
            vary_values=True,
        )
        if per_channel_qparams
        else create_per_tensor_qparams(builder)
    )

    pool = builder.call_operator(
        op=exir_ops.edge.fused_quant.max_pool2d_with_indices.default,
        args=(
            inp_placeholder,
            *inp_qparams,
            *out_qparams,
            *pool_args,
        ),
    )
    values = builder.call_getitem(pool, 0)
    indices = builder.call_getitem(pool, 1)
    builder.output([values, indices])
    return builder.get_program(), inp


def _build_fused_quant_avg_pool2d_graph(
    input_shape: tuple[int, ...],
    pool_args: tuple[Any, ...],
    per_channel_qparams: bool = False,
) -> tuple[ExportedProgram, torch.Tensor]:
    inp = torch.randn(*input_shape).to(torch.int8)
    builder = ProgramBuilder()
    inp_placeholder = builder.placeholder("inp", inp)
    inp_qparams = (
        create_per_axis_qparams(
            builder,
            input_shape[1],
            len(input_shape),
            axis=1,
            dtype=torch.float32,
            vary_values=True,
        )
        if per_channel_qparams
        else create_per_tensor_qparams(builder, dtype=torch.float32)
    )
    out_qparams = (
        create_per_axis_qparams(
            builder,
            input_shape[1],
            len(input_shape),
            axis=1,
            vary_values=True,
        )
        if per_channel_qparams
        else create_per_tensor_qparams(builder)
    )
    pool = builder.call_operator(
        op=exir_ops.edge.fused_quant.avg_pool2d.default,
        args=(inp_placeholder, *inp_qparams, *out_qparams, *pool_args),
    )
    builder.output([pool])
    return builder.get_program(), inp


class ToChannelsLastTest(unittest.TestCase):
    """Tests for the ToChannelsLast optimization pass."""

    def assert_bit_exact(self, before: object, after: object) -> None:
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_pointwise_conv_with_singleton_weight_dims_is_bit_exact(self) -> None:
        indices = torch.arange(64)
        inp_vector = ((indices * 37 + 17) % 256).to(torch.uint8)
        weight_vector = ((indices * 53 + 29) % 256 - 128).to(torch.int8)
        inp = inp_vector.reshape(1, 64, 1, 1).expand(1, 64, 7, 46).contiguous()
        weight = weight_vector.reshape(1, 64, 1, 1).expand(16, 64, 1, 1).contiguous()
        weight_ohwi = weight.permute(0, 2, 3, 1).clone(
            memory_format=torch.contiguous_format
        )
        roundtripped_weight = weight_ohwi.permute(0, 3, 1, 2).contiguous()
        self.assertTrue(roundtripped_weight.is_contiguous())
        self.assertEqual(roundtripped_weight.stride(), (64, 1, 64, 64))
        self.assertEqual(
            roundtripped_weight.clone(memory_format=torch.contiguous_format).stride(),
            (64, 1, 1, 1),
        )
        bias = torch.full((16,), -0.28)
        inp_scale = torch.tensor(0.01256)
        inp_zero_point = torch.tensor(124)
        weight_scales = torch.full((16, 1, 1, 1), 0.00965)
        weight_zero_points = torch.zeros((16, 1, 1, 1), dtype=torch.int64)
        inp_qparams = (inp_scale, inp_zero_point, torch.float32, 0, 255)
        weight_qparams = (
            weight_scales,
            weight_zero_points,
            torch.float32,
            -128,
            127,
        )
        bias_qparams = (None, None, torch.float32, -2147483648, 2147483647)
        out_qparams = (None, None, torch.float32, -2147483648, 2147483647)
        conv_args = ([1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        args = (
            inp,
            weight,
            bias,
            *inp_qparams,
            *weight_qparams,
            *bias_qparams,
            *out_qparams,
            *conv_args,
        )

        builder = ProgramBuilder()
        graph_args: list[object] = []
        runtime_inputs: list[torch.Tensor] = []
        for index, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                graph_args.append(builder.placeholder(f"arg_{index}", value))
                runtime_inputs.append(value)
            else:
                graph_args.append(value)
        conv = builder.call_operator(
            exir_ops.edge.fused_quant.convolution.default,
            args=tuple(graph_args),
        )
        builder.output([conv])
        program = builder.get_program()
        [baseline] = program.module()(*runtime_inputs)
        ToChannelsLast().call(program.graph_module)
        [converted] = program.module()(*runtime_inputs)
        self.assert_bit_exact(baseline, converted)

    @parameterized.expand(
        [
            # Conv1d with and without bias
            (
                "conv1d_with_bias",
                (1, 4, 8),
                (2, 4, 3),
                (2,),
                _CONVOLUTION1D_ARGS,
            ),
            (
                "conv1d_no_bias",
                (1, 4, 8),
                (2, 4, 3),
                None,
                _CONVOLUTION1D_ARGS,
            ),
            # Conv2d with and without bias
            (
                "conv2d_with_bias",
                (1, 4, 8, 8),
                (2, 4, 3, 3),
                (2,),
                _CONVOLUTION2D_ARGS,
            ),
            (
                "conv2d_no_bias",
                (1, 4, 8, 8),
                (2, 4, 3, 3),
                None,
                _CONVOLUTION2D_ARGS,
            ),
            # Conv3d with and without bias
            (
                "conv3d_with_bias",
                (1, 4, 8, 8, 8),
                (2, 4, 3, 3, 3),
                (2,),
                _CONVOLUTION3D_ARGS,
            ),
            (
                "conv3d_no_bias",
                (1, 4, 8, 8, 8),
                (2, 4, 3, 3, 3),
                None,
                _CONVOLUTION3D_ARGS,
            ),
        ]
    )
    def test_conv_to_channels_last(
        self,
        _name: str,
        input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        bias_shape: tuple[int, ...] | None,
        conv_args: tuple[Any, ...],
    ) -> None:
        """
        Test that ToChannelsLast correctly transforms fused_quant.convolution
        to fused_quant.convolution_channels_last with appropriate permutes.

        The expected pattern after optimization is:
        permute(NCHW->NHWC) -> permute(OIHW->OHWI) ->
        fused_quant.convolution_channels_last -> permute(NHWC->NCHW)
        """
        program, example_inputs = _build_fused_quant_conv_graph(
            input_shape, weight_shape, bias_shape, conv_args
        )
        before = program.module()(*example_inputs)

        # Apply the optimization pass
        result = ToChannelsLast().call(program.graph_module)
        after = program.module()(*example_inputs)

        self.assertTrue(result.modified)
        self.assert_bit_exact(before, after)

        # Verify the graph structure
        graph = result.graph_module.graph

        # We should have:
        # 1. permute_copy (input NCHW -> NHWC)
        # 2. permute_copy (weight OIHW -> OHWI)
        # 3. permute_copy (weight scale OIHW -> OHWI)
        # 4. permute_copy (weight zero point OIHW -> OHWI)
        # 5. fused_quant.convolution_channels_last
        # 6. permute_copy (output NHWC -> NCHW)
        permute_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        )
        conv_channels_last_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution_channels_last.default,
        )

        self.assertEqual(
            len(permute_nodes), 5, "Expected 5 tensor/qparam permute nodes"
        )
        self.assertEqual(
            len(conv_channels_last_nodes),
            1,
            "Expected 1 convolution_channels_last node",
        )

        # Verify no original convolution nodes remain
        conv_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution.default,
        )
        self.assertEqual(
            len(conv_nodes), 0, "Original convolution node should be removed"
        )

    def test_conv_to_channels_last_permutes_activation_qparams(self) -> None:
        program, example_inputs = _build_fused_quant_conv_graph(
            (1, 4, 8, 8),
            (6, 4, 3, 3),
            None,
            _CONVOLUTION2D_ARGS,
            per_channel_activations=True,
        )
        before = program.module()(*example_inputs)

        result = ToChannelsLast().call(program.graph_module)
        self.assert_bit_exact(before, program.module()(*example_inputs))
        graph = result.graph_module.graph
        [conv] = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution_channels_last.default,
        )
        self.assertEqual(
            tuple(get_arg(conv, "inp_scale", torch.fx.Node).meta["val"].shape),
            (1, 1, 1, 4),
        )
        self.assertEqual(
            tuple(get_arg(conv, "weight_scale", torch.fx.Node).meta["val"].shape),
            (6, 1, 1, 1),
        )
        self.assertEqual(
            tuple(get_arg(conv, "out_scale", torch.fx.Node).meta["val"].shape),
            (1, 1, 1, 6),
        )

    def test_conv_to_channels_last_preserves_transposed_arguments(self) -> None:
        program, example_inputs = _build_fused_quant_conv_graph(
            (1, 2, 4, 4),
            (2, 3, 3, 3),
            None,
            ([2, 2], [1, 1], [1, 1], True, [1, 1], 1),
        )
        before = program.module()(*example_inputs)

        result = ToChannelsLast().call(program.graph_module)

        self.assertTrue(result.modified)
        self.assert_bit_exact(before, program.module()(*example_inputs))
        [conv] = result.graph_module.graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution_channels_last.default,
        )
        self.assertTrue(get_arg(conv, "transposed", bool))
        self.assertEqual(get_arg(conv, "output_padding", list[int]), [1, 1])

    def test_max_pool2d_with_indices_to_channels_last(self) -> None:
        pool_args: tuple[Any, ...] = (
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
            True,
        )
        program, inp = _build_fused_quant_max_pool2d_with_indices_graph(
            (1, 4, 8, 8), pool_args
        )
        before = program.module()(inp)

        result = ToChannelsLast().call(program.graph_module)
        self.assertTrue(result.modified)
        self.assert_bit_exact(before, program.module()(inp))
        graph = result.graph_module.graph

        permute_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        )
        self.assertEqual(len(permute_nodes), 3)
        self.assertEqual(permute_nodes[0].args[1], [0, 2, 3, 1])
        self.assertEqual(permute_nodes[1].args[1], [0, 3, 1, 2])
        self.assertEqual(permute_nodes[2].args[1], [0, 3, 1, 2])

        pool_nodes = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.max_pool2d_with_indices_channels_last.default,
        )
        self.assertEqual(len(pool_nodes), 1)
        pool = pool_nodes[0]
        self.assertEqual(get_arg(pool, "kernel_size"), pool_args[0])
        self.assertEqual(get_arg(pool, "stride"), pool_args[1])
        self.assertEqual(get_arg(pool, "padding"), pool_args[2])
        self.assertEqual(get_arg(pool, "dilation"), pool_args[3])
        self.assertEqual(get_arg(pool, "ceil_mode"), pool_args[4])
        self.assertEqual(
            [tuple(value.shape) for value in pool.meta["val"]],
            [(1, 5, 5, 4), (1, 5, 5, 4)],
        )
        getitems = graph.find_nodes(op="call_function", target=operator.getitem)
        self.assertEqual({node.args[1] for node in getitems}, {0, 1})

        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.fused_quant.max_pool2d_with_indices.default,
                )
            ),
            0,
        )

    def test_max_pool2d_with_indices_permutes_qparams(self) -> None:
        program, inp = _build_fused_quant_max_pool2d_with_indices_graph(
            (1, 4, 8, 8),
            ([3, 3], [2, 2], [1, 1], [1, 1], True),
            per_channel_qparams=True,
        )
        before = program.module()(inp)

        result = ToChannelsLast().call(program.graph_module)
        self.assert_bit_exact(before, program.module()(inp))
        graph = result.graph_module.graph
        [pool] = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.max_pool2d_with_indices_channels_last.default,
        )
        self.assertEqual(
            tuple(get_arg(pool, "inp_scale", torch.fx.Node).meta["val"].shape),
            (1, 1, 1, 4),
        )
        self.assertEqual(
            tuple(get_arg(pool, "out_scale", torch.fx.Node).meta["val"].shape),
            (1, 1, 1, 4),
        )

    def test_avg_pool2d_to_channels_last(self) -> None:
        pool_args: tuple[Any, ...] = (
            [3, 3],
            [],
            [1, 1],
            False,
            False,
            None,
        )
        program, inp = _build_fused_quant_avg_pool2d_graph((1, 4, 8, 8), pool_args)
        before = program.module()(inp)
        result = ToChannelsLast().call(program.graph_module)
        self.assert_bit_exact(before, program.module()(inp))
        graph = result.graph_module.graph

        permutes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        )
        self.assertEqual(
            [node.args[1] for node in permutes], [[0, 2, 3, 1], [0, 3, 1, 2]]
        )
        [pool] = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
        )
        self.assertEqual(get_arg(pool, "kernel_size"), pool_args[0])
        self.assertEqual(get_arg(pool, "stride"), pool_args[1])
        self.assertEqual(get_arg(pool, "padding"), pool_args[2])
        self.assertEqual(get_arg(pool, "ceil_mode"), pool_args[3])
        self.assertEqual(get_arg(pool, "count_include_pad"), pool_args[4])
        self.assertEqual(get_arg(pool, "divisor_override"), pool_args[5])
        self.assertEqual(tuple(pool.meta["val"].shape), (1, 3, 3, 4))
        self.assertEqual(
            len(
                graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.fused_quant.avg_pool2d.default,
                )
            ),
            0,
        )

    def test_avg_pool2d_permutes_qparams(self) -> None:
        program, inp = _build_fused_quant_avg_pool2d_graph(
            (1, 4, 8, 8),
            ([3, 3], [2, 2], [1, 1], False, True, None),
            per_channel_qparams=True,
        )
        before = program.module()(inp)
        result = ToChannelsLast().call(program.graph_module)
        self.assert_bit_exact(before, program.module()(inp))
        graph = result.graph_module.graph
        [pool] = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.avg_pool2d_channels_last.default,
        )
        self.assertEqual(
            tuple(get_arg(pool, "inp_scale", torch.fx.Node).meta["val"].shape),
            (1, 1, 1, 4),
        )
        self.assertEqual(
            tuple(get_arg(pool, "out_scale", torch.fx.Node).meta["val"].shape),
            (1, 1, 1, 4),
        )
