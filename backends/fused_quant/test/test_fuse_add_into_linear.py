# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Any

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.graph_utils import get_constant
from executorch.backends.fused_quant.optimization_passes.fuse_add_into_linear import (
    FuseAddIntoLinear,
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
from torch.export.graph_signature import InputKind


class FuseAddIntoLinearTest(unittest.TestCase):
    def _build_affine_add_scalar(
        self, target: Any, with_bias: bool
    ) -> tuple[ExportedProgram, torch.Tensor]:
        builder = ProgramBuilder()
        out_channels = 4

        if target == exir_ops.edge.fused_quant.linear.default:
            inp_value = torch.randn(2, 8)
            weight_value = torch.randint(-16, 16, (out_channels, 8), dtype=torch.int8)
        elif target == exir_ops.edge.fused_quant.convolution.default:
            inp_value = torch.randn(1, 3, 5, 5)
            weight_value = torch.randint(
                -16, 16, (out_channels, 3, 1, 1), dtype=torch.int8
            )
        else:
            inp_value = torch.randn(1, 5, 5, 3)
            weight_value = torch.randint(
                -16, 16, (out_channels, 1, 1, 3), dtype=torch.int8
            )

        bias_value = torch.arange(out_channels, dtype=torch.float32)
        expected_bias = bias_value + 0.5 if with_bias else torch.full((4,), 0.5)

        inp = builder.placeholder("x", inp_value)
        weight = builder.placeholder(
            "weight", weight_value, input_kind=InputKind.BUFFER
        )
        bias = (
            builder.placeholder("bias", bias_value, input_kind=InputKind.BUFFER)
            if with_bias
            else None
        )
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_channel_qparams(
            builder, out_channels, dtype=torch.float32
        )
        affine_out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.int8
        )
        affine_args: tuple[Any, ...] = (
            inp,
            weight,
            bias,
            *inp_qparams,
            *weight_qparams,
            None,
            None,
            torch.float32,
            0,
            0,
            *affine_out_qparams,
        )
        if target != exir_ops.edge.fused_quant.linear.default:
            affine_args = (
                *affine_args,
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
            )

        affine = builder.call_operator(op=target, args=affine_args)
        add_inp_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.float32
        )
        add_out_qparams = create_per_tensor_qparams(
            builder, scale=0.125, dtype=torch.int8
        )
        add = builder.call_operator(
            op=exir_ops.edge.fused_quant.add.Scalar,
            args=(affine, *add_inp_qparams, *add_out_qparams, 2.0, 0.25),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(add, 0.125, 0, -128, 127, torch.int8),
        )
        builder.output([dequantize])
        return builder.get_program(), expected_bias

    @parameterized.expand(
        [
            ("linear_bias", exir_ops.edge.fused_quant.linear.default, True),
            ("linear_no_bias", exir_ops.edge.fused_quant.linear.default, False),
            ("conv_bias", exir_ops.edge.fused_quant.convolution.default, True),
            ("conv_no_bias", exir_ops.edge.fused_quant.convolution.default, False),
            (
                "conv_channels_last_bias",
                exir_ops.edge.fused_quant.convolution_channels_last.default,
                True,
            ),
        ]
    )
    def test_scalar_add_folded_into_bias(
        self, _name: str, target: Any, with_bias: bool
    ) -> None:
        ep, expected_bias = self._build_affine_add_scalar(target, with_bias)

        result = FuseAddIntoLinear().call(ep)

        self.assertTrue(result.modified)
        graph = result.exported_program.graph_module.graph
        self.assertEqual(
            graph.find_nodes(
                op="call_function", target=exir_ops.edge.fused_quant.add.Scalar
            ),
            [],
        )
        affine_node = graph.find_nodes(op="call_function", target=target)[0]
        bias_node = get_arg(affine_node, "bias", torch.fx.Node)
        actual_bias = get_constant(result.exported_program, bias_node)
        torch.testing.assert_close(actual_bias, expected_bias)

    def test_scalar_add_not_folded_when_qparams_mismatch(self) -> None:
        builder = ProgramBuilder()
        out_channels = 4
        inp_value = torch.randn(2, 8)
        weight_value = torch.randint(-16, 16, (out_channels, 8), dtype=torch.int8)
        bias_value = torch.arange(out_channels, dtype=torch.float32)

        inp = builder.placeholder("x", inp_value)
        weight = builder.placeholder(
            "weight", weight_value, input_kind=InputKind.BUFFER
        )
        bias = builder.placeholder("bias", bias_value, input_kind=InputKind.BUFFER)
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_channel_qparams(
            builder, out_channels, dtype=torch.float32
        )
        affine_out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.int8
        )
        affine = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                inp,
                weight,
                bias,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.float32,
                0,
                0,
                *affine_out_qparams,
            ),
        )
        # Mismatched inp_scale (0.25) vs affine out_scale (0.5)
        add_inp_qparams = create_per_tensor_qparams(
            builder, scale=0.25, dtype=torch.float32
        )
        add_out_qparams = create_per_tensor_qparams(
            builder, scale=0.125, dtype=torch.int8
        )
        add = builder.call_operator(
            op=exir_ops.edge.fused_quant.add.Scalar,
            args=(affine, *add_inp_qparams, *add_out_qparams, 2.0, 0.25),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(add, 0.125, 0, -128, 127, torch.int8),
        )
        builder.output([dequantize])
        ep = builder.get_program()

        result = FuseAddIntoLinear().call(ep)

        self.assertFalse(result.modified)

    def test_scalar_add_folded_when_qparams_are_none(self) -> None:
        builder = ProgramBuilder()
        out_channels = 4
        inp_value = torch.randn(2, 8)
        weight_value = torch.randint(-16, 16, (out_channels, 8), dtype=torch.int8)
        bias_value = torch.arange(out_channels, dtype=torch.float32)
        expected_bias = bias_value + 0.5

        inp = builder.placeholder("x", inp_value)
        weight = builder.placeholder(
            "weight", weight_value, input_kind=InputKind.BUFFER
        )
        bias = builder.placeholder("bias", bias_value, input_kind=InputKind.BUFFER)
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_channel_qparams(
            builder, out_channels, dtype=torch.float32
        )
        # Affine out_scale and out_zero_point are None
        affine = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                inp,
                weight,
                bias,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.float32,
                0,
                0,
                None,
                None,
                torch.int8,
                -128,
                127,
            ),
        )
        # Add inp_scale and inp_zero_point are also None
        add = builder.call_operator(
            op=exir_ops.edge.fused_quant.add.Scalar,
            args=(
                affine,
                None,
                None,
                torch.float32,
                -128,
                127,
                None,
                None,
                torch.int8,
                -128,
                127,
                2.0,
                0.25,
            ),
        )
        builder.output([add])
        ep = builder.get_program()

        result = FuseAddIntoLinear().call(ep)

        self.assertTrue(result.modified)
        graph = result.exported_program.graph_module.graph
        self.assertEqual(
            graph.find_nodes(
                op="call_function", target=exir_ops.edge.fused_quant.add.Scalar
            ),
            [],
        )
        affine_node = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )[0]
        bias_node = get_arg(affine_node, "bias", torch.fx.Node)
        actual_bias = get_constant(result.exported_program, bias_node)
        torch.testing.assert_close(actual_bias, expected_bias)

    def test_scalar_add_not_folded_when_dtype_changes(self) -> None:
        builder = ProgramBuilder()
        out_channels = 4
        inp_value = torch.randn(2, 8)
        weight_value = torch.randint(-16, 16, (out_channels, 8), dtype=torch.int8)
        bias_value = torch.arange(out_channels, dtype=torch.float32)

        inp = builder.placeholder("x", inp_value)
        weight = builder.placeholder(
            "weight", weight_value, input_kind=InputKind.BUFFER
        )
        bias = builder.placeholder("bias", bias_value, input_kind=InputKind.BUFFER)
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_channel_qparams(
            builder, out_channels, dtype=torch.float32
        )
        affine_out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.int8
        )
        affine = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                inp,
                weight,
                bias,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.float32,
                0,
                0,
                *affine_out_qparams,
            ),
        )
        add_inp_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.float32
        )
        # out_dtype=int16 differs from affine out_dtype=int8
        add_out_qparams = create_per_tensor_qparams(
            builder, scale=0.125, dtype=torch.int16
        )
        add = builder.call_operator(
            op=exir_ops.edge.fused_quant.add.Scalar,
            args=(affine, *add_inp_qparams, *add_out_qparams, 2.0, 0.25),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(add, 0.125, 0, -32768, 32767, torch.int16),
        )
        builder.output([dequantize])
        ep = builder.get_program()

        result = FuseAddIntoLinear().call(ep)

        self.assertFalse(result.modified)

    def test_scalar_add_folded_through_channels_last_permute(self) -> None:
        builder = ProgramBuilder()
        out_channels = 4
        # NHWC input for channels-last conv
        inp_value = torch.randn(1, 5, 5, 3)
        weight_value = torch.randint(-16, 16, (out_channels, 1, 1, 3), dtype=torch.int8)
        bias_value = torch.arange(out_channels, dtype=torch.float32)
        expected_bias = bias_value + 0.5

        inp = builder.placeholder("x", inp_value)
        weight = builder.placeholder(
            "weight", weight_value, input_kind=InputKind.BUFFER
        )
        bias = builder.placeholder("bias", bias_value, input_kind=InputKind.BUFFER)
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_channel_qparams(
            builder, out_channels, dtype=torch.float32, weight_ndim=4
        )
        affine_out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.int8
        )
        affine = builder.call_operator(
            op=exir_ops.edge.fused_quant.convolution_channels_last.default,
            args=(
                inp,
                weight,
                bias,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.float32,
                0,
                0,
                *affine_out_qparams,
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
            ),
        )
        # NHWC [1,5,5,4] -> NCHW [1,4,5,5]
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(affine, [0, 3, 1, 2]),
        )
        add_inp_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.float32
        )
        add_out_qparams = create_per_tensor_qparams(
            builder, scale=0.125, dtype=torch.int8
        )
        add = builder.call_operator(
            op=exir_ops.edge.fused_quant.add.Scalar,
            args=(permute, *add_inp_qparams, *add_out_qparams, 2.0, 0.25),
        )
        dequantize = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(add, 0.125, 0, -128, 127, torch.int8),
        )
        builder.output([dequantize])
        ep = builder.get_program()

        result = FuseAddIntoLinear().call(ep)

        self.assertTrue(result.modified)
        graph = result.exported_program.graph_module.graph
        self.assertEqual(
            graph.find_nodes(
                op="call_function", target=exir_ops.edge.fused_quant.add.Scalar
            ),
            [],
        )
        affine_node = graph.find_nodes(
            op="call_function",
            target=exir_ops.edge.fused_quant.convolution_channels_last.default,
        )[0]
        bias_node = get_arg(affine_node, "bias", torch.fx.Node)
        actual_bias = get_constant(result.exported_program, bias_node)
        torch.testing.assert_close(actual_bias, expected_bias)

    def test_scalar_add_not_folded_with_per_channel_qparams_through_passthrough(
        self,
    ) -> None:
        builder = ProgramBuilder()
        out_channels = 4
        inp_value = torch.randint(-16, 16, (2, 8), dtype=torch.int8)
        weight_value = torch.randint(-16, 16, (out_channels, 8), dtype=torch.int8)
        bias_value = torch.arange(out_channels, dtype=torch.float32)

        inp = builder.placeholder("x", inp_value)
        weight = builder.placeholder(
            "weight", weight_value, input_kind=InputKind.BUFFER
        )
        bias = builder.placeholder("bias", bias_value, input_kind=InputKind.BUFFER)
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.float32)
        weight_qparams = create_per_channel_qparams(
            builder, out_channels, dtype=torch.float32
        )
        affine_out_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.int8
        )
        affine = builder.call_operator(
            op=exir_ops.edge.fused_quant.linear.default,
            args=(
                inp,
                weight,
                bias,
                *inp_qparams,
                *weight_qparams,
                None,
                None,
                torch.float32,
                0,
                0,
                *affine_out_qparams,
            ),
        )
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(affine, [0, 1]),
        )
        add_inp_qparams = create_per_tensor_qparams(
            builder, scale=0.5, dtype=torch.float32
        )
        add_out_qparams = create_per_axis_qparams(
            builder,
            out_channels,
            ndim=2,
            axis=1,
        )
        add = builder.call_operator(
            op=exir_ops.edge.fused_quant.add.Scalar,
            args=(permute, *add_inp_qparams, *add_out_qparams, 0.0, 1.0),
        )
        builder.output([add])
        ep = builder.get_program()

        result = FuseAddIntoLinear().call(ep)

        self.assertFalse(result.modified)

    def test_scalar_add_not_folded_when_affine_has_multiple_users(self) -> None:
        ep, _ = self._build_affine_add_scalar(
            exir_ops.edge.fused_quant.linear.default, True
        )
        graph = ep.graph_module.graph
        linear = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.linear.default
        )[0]
        output = graph.find_nodes(op="output")[0]
        with graph.inserting_before(output):
            extra_user = graph.call_function(
                exir_ops.edge.aten.clone.default, args=(linear,)
            )
            extra_user.meta["val"] = linear.meta["val"]
        output.args = ([output.args[0][0], extra_user],)
        ep.graph_module.recompile()

        result = FuseAddIntoLinear().call(ep)

        self.assertFalse(result.modified)
