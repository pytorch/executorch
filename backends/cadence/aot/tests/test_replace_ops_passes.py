# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
import unittest
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Union

import torch
from executorch.backends.cadence.aot.graph_builder import (
    GraphBuilder,
    single_op_builder,
)
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.backends.cadence.aot.replace_ops import (
    ForceChannelLastForConvPass,
    MakeSliceAndCatDimOutermostPass,
    ReplaceAddMMWithLinearPass,
    ReplaceAtenApproxGeluWithApproxGeluPass,
    ReplaceAtenConvolutionWithJarvisConvolutionPass,
    ReplaceConstantPadNdWithSlicePass,
    ReplaceConvolutionOptionalArgsWithConcreteArgsPass,
    ReplaceConvWithIm2RowAndLinear,
    ReplaceEmptyTensorsWithFullPass,
    ReplaceFunctionallyEquivalentOpTargets,
    ReplaceIm2RowWithViewPass,
    ReplaceLinearWithFullyConnectedOpPass,
    ReplaceMatmulWithTransposedMatmulPass,
    ReplaceMMWithAddMMPass,
    ReplaceNopTransposeOrPermuteWithViewPass,
    ReplacePadWithCatPass,
    ReplacePermuteWithTransposePass,
    ReplacePowWithMulPass,
    ReplaceRepeatWithCatPass,
    ReplaceScalarTensorWithFullPass,
    ReplaceScalarWithTensorArgPass,
    ReplaceSelectWithViewOpPass,
    ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass,
    ReplaceSplitWithSlicePass,
    ReplaceSqueezeAndUnsqueezeWithViewPass,
    ReplaceTransposedConvWithLinearPass,
    ReplaceTrivialConvWithLinear,
    ReplaceWhereWithFullArgsWithWhereScalar,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes import dead_code_elimination_pass

from parameterized.parameterized import parameterized
from torch.fx.passes.infra.pass_base import PassResult


class TestReplaceOpsPasses(unittest.TestCase):
    def assertTargetCountEqual(
        self,
        graph_module: torch.fx.GraphModule,
        target: Union[Callable[..., Any], str],
        expected_count: int,
    ):
        """Helper function to check the number of nodes with a given target."""
        actual_count = count_node(graph_module, target)
        self.assertEqual(
            actual_count,
            expected_count,
            f"{target} count mismatch for graph {graph_module}",
        )

    def assertTargetCountsEqual(
        self,
        graph_module: torch.fx.GraphModule,
        targets_and_counts: List[Tuple[Union[Callable[..., Any], str], int]],
    ):
        """Helper function to check the number of nodes of all types for a given target."""
        for target, expected_count in targets_and_counts:
            self.assertTargetCountEqual(graph_module, target, expected_count)

    @parameterized.expand(
        [
            (
                "regular",
                (64, 33),  # x_shape
                (33, 128),  # y_shape
            ),
            (
                "batched",
                (2, 48, 48),  # x_shape
                (2, 48, 48),  # y_shape
            ),
        ],
    )
    @torch.no_grad()
    def test_replace_matmul_with_transposed_matmul(
        self,
        _,
        x_shape: Tuple[int],
        y_shape: Tuple[int],
    ) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*x_shape, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn(*y_shape, dtype=torch.float32))
        matmul = builder.call_operator(
            op=exir_ops.edge.cadence.quantized_matmul.default,
            args=(
                x,
                0,  # X_zero_point
                y,
                0,  # Y_zero_point,
                None,  # bias
                1,  # out_multiplier
                0,  # out_shift
                0,  # out_zero_point
                False,  # transposed=False
            ),
        )
        builder.output([matmul])
        original_gm = builder.get_graph_module()
        p = ReplaceMatmulWithTransposedMatmulPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int),
            1,
        )
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.quantized_matmul.default
            ),
            1,
        )

    @parameterized.expand(
        [
            ("2d", (3, 5), [0, 0]),  # shape  # padding
            ("3d", (20, 1, 80), [0, 0, 0]),  # shape  # padding
        ],
    )
    @torch.no_grad()
    def test_replace_constant_pad_nd_with_slice(
        self, _, shape: Tuple[int], padding: Tuple[int]
    ):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        matmul = builder.call_operator(
            op=exir_ops.edge.aten.constant_pad_nd.default,
            args=(x, [0, 0, 0, 0]),
        )
        builder.output([matmul])
        original_gm = builder.get_graph_module()
        p = ReplaceConstantPadNdWithSlicePass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.slice.Tensor),
            1,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.constant_pad_nd.default),
            0,
        )

    @parameterized.expand(
        [
            ["3d", (7, 5, 6), 1.23],
            ["2d", (7, 5), 2],
            ["1d", (10,), 42949],
        ]
    )
    @torch.no_grad()
    def test_add_replace_scalar_with_tensor_arg(
        self, _, shape: Tuple[int], other: float
    ):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.add.Scalar,
            args=(x, other),
        )
        p = ReplaceScalarWithTensorArgPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.add.Tensor),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.add.Scalar),
            0,
        )

    @parameterized.expand(
        [
            ["3d", (7, 5, 6), 1.23],
            ["2d", (7, 5), 2],
            ["1d", (10,), 42949],
        ]
    )
    @torch.no_grad()
    def test_sub_replace_scalar_with_tensor_arg(
        self, _, shape: Tuple[int], other: float
    ):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.sub.Scalar,
            args=(x, other),
        )
        p = ReplaceScalarWithTensorArgPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.sub.Tensor),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.sub.Scalar),
            0,
        )

    @parameterized.expand(
        [
            ["3d", (7, 5, 6), 1.23],
            ["2d", (7, 5), 2],
            ["1d", (10,), 42949],
        ]
    )
    @torch.no_grad()
    def test_mul_replace_scalar_with_tensor_arg(
        self, _, shape: Tuple[int], other: float
    ):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.mul.Scalar,
            args=(x, other),
        )
        p = ReplaceScalarWithTensorArgPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mul.Tensor),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mul.Scalar),
            0,
        )

    @parameterized.expand(
        [
            ["3d", (7, 5, 6), 1.23],
            ["2d", (7, 5), 2],
            ["1d", (10,), 42949],
        ]
    )
    @torch.no_grad()
    def test_div_replace_scalar_with_tensor_arg(
        self,
        _,
        shape: Tuple[int],
        other: float,
    ):
        x = torch.randn(*shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.div.Scalar,
            args=(x, other),
        )
        p = ReplaceScalarWithTensorArgPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.div.Tensor),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.div.Scalar),
            0,
        )

    @parameterized.expand(
        [
            ["4d", (2, 3, 5, 6)],
            ["3d", (7, 6, 5)],
            ["2d", (4, 4)],
            ["1d", (316)],
        ]
    )
    @torch.no_grad()
    def test_replace_functionally_equivalent_op_targets_relu(
        self, _, shape: Tuple[int]
    ):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.relu_.default,
            args=(x,),
        )
        p = ReplaceFunctionallyEquivalentOpTargets()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.relu.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.relu_.default),
            0,
        )

    @parameterized.expand(
        [["split_linear_tensor", (50,), i, 0] for i in range(2, 7)]
        + [["split_leading_dim", (10, 2, 3), i, 0] for i in range(2, 7)]
        + [["split_trailing_dim", (3, 3, 6), i, 2] for i in range(2, 6)]
        + [["split_middle_dim", (3, 5, 14, 2, 3), i, 2] for i in range(2, 7)]
    )
    @torch.no_grad()
    def test_replace_functionally_equivalent_op_targets_unsafe_split(
        self, _, shape: Tuple[int], split_size: int, dim: int
    ):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.unsafe_split.Tensor,
            args=(x, split_size, dim),
        )
        p = ReplaceFunctionallyEquivalentOpTargets()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.split_copy.Tensor),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.unsafe_split.Tensor), 0, x
        )

    @parameterized.expand(
        [
            [(1, 8, 33), 8, 16, 3],
            [(1, 8, 33), 8, 16, 5, 2],
            [(1, 8, 33), 8, 16, 3, 2, 4, 3, False, False, False],
            # # channel last
            [(1, 33, 8), 8, 16, 3, 1, 0, 1, False, False, True],
            [(1, 33, 8), 8, 16, 5, 2, 0, 1, False, True, True],
        ]
    )
    @torch.no_grad()
    def test_replace_transposed_conv_with_linear(
        self,
        shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        depthwise: bool = False,
        bias_enabled: bool = True,
        channel_last: bool = False,
    ):
        transposed = True
        output_padding = [0]
        groups = in_channels if depthwise else 1
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        weights = builder.placeholder(
            "weights",
            torch.randn([in_channels, out_channels, kernel], dtype=torch.float32),
        )
        bias = (
            builder.placeholder(
                "bias", torch.randn([out_channels], dtype=torch.float32)
            )
            if bias_enabled
            else None
        )
        if channel_last:
            x = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(x, [0, 2, 1]),
            )
        convolution = builder.call_operator(
            op=exir_ops.edge.aten.convolution.default,
            args=(
                x,
                weights,
                bias,
                [stride],
                [padding],
                [dilation],
                transposed,
                output_padding,
                groups,
            ),
        )
        if channel_last:
            convolution = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(convolution, [0, 2, 1]),
            )
        builder.output([convolution])
        original_gm = builder.get_graph_module()

        p1 = ReplaceAtenConvolutionWithJarvisConvolutionPass()
        p2 = ReplaceTransposedConvWithLinearPass()
        graph_after_passes = cast(
            PassResult, p2(cast(PassResult, p1(original_gm)).graph_module)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.convolution.default),
            0,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.convolution.default),
            0,
        )

    @parameterized.expand(
        [
            [(1, 8, 33), 8, 16, 3, 2, 4, 3, False, False, False],
            # # depthwise
            [(1, 8, 33), 8, 16, 3, 1, 0, 1, True, False, False],
            [(1, 8, 33), 8, 16, 3, 2, 4, 3, True, False, False],
            # channel last (uses a permute op before calling conv1d)
            [(1, 33, 8), 8, 16, 3, 1, 0, 1, False, False, True],
            [(1, 33, 8), 8, 16, 3, 2, 4, 3, True, False, True],
        ]
    )
    @torch.no_grad()
    def test_replace_convolution_optional_args_with_concrete_args(
        self,
        shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        depthwise: bool = False,
        bias_enabled: bool = True,
        channel_last: bool = False,
    ):
        transposed = True
        output_padding = [0]
        groups = in_channels if depthwise else 1
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        weights = builder.placeholder(
            "weights",
            torch.randn([in_channels, out_channels, kernel], dtype=torch.float32),
        )
        bias = (
            builder.placeholder(
                "bias", torch.randn([out_channels], dtype=torch.float32)
            )
            if bias_enabled
            else None
        )
        if channel_last:
            x = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(x, [0, 2, 1]),
            )
        convolution = builder.call_operator(
            op=exir_ops.edge.aten.convolution.default,
            args=(
                x,
                weights,
                bias,
                [stride],
                [padding],
                [dilation],
                transposed,
                output_padding,
                groups,
            ),
        )
        if channel_last:
            convolution = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(convolution, [0, 2, 1]),
            )
        builder.output([convolution])
        original_gm = builder.get_graph_module()
        p = ReplaceConvolutionOptionalArgsWithConcreteArgsPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.convolution.default),
            1,
        )

    @parameterized.expand(
        [
            [(1, 2, 3), [1, 1]],
            [
                (20, 1, 80),
                [1, 4],
            ],
        ]
    )
    @torch.no_grad()
    def test_replace_pad_with_cat(self, shape: Tuple[int], padding: Tuple[int]):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.constant_pad_nd.default,
            args=(x, padding),
        )
        p = ReplacePadWithCatPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.constant_pad_nd.default),
            0,
        )

    @torch.no_grad()
    def test_replace_repeat_with_cat(self):
        x = torch.randn([3, 5])
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.repeat.default,
            args=(x, [1, 2]),
        )
        p = ReplaceRepeatWithCatPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.repeat.default),
            0,
        )

    @parameterized.expand(
        [
            # x, mask
            [(1,)],
            [(3, 4)],
            [(7, 8, 3)],
            [(3, 3, 2, 4)],
            [(36, 1, 2, 80), (1,)],
            # tests where mask will be broadcasted
            [(36, 1, 2, 80), (1, 1, 2, 1)],
            [(36, 2, 8, 4), (36, 1, 1, 4)],
            [(36, 2, 8, 4), (2, 1, 4)],
        ]
    )
    @torch.no_grad()
    def test_replace_masked_scalar_tensor_with_full(
        self,
        shape: Tuple[int],
        mask_shape: Union[Tuple[int, ...], None] = None,
    ):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        mask = builder.placeholder(
            "mask",
            torch.randint(0, 2, mask_shape if mask_shape else shape, dtype=torch.bool),
        )
        scalar_tensor = builder.call_operator(
            op=exir_ops.edge.aten.scalar_tensor.default,
            args=(0.123,),
            kwargs={
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
            },
        )
        aten_where_self = builder.call_operator(
            op=exir_ops.edge.aten.where.self,
            args=(mask, scalar_tensor, x),
        )
        builder.output([aten_where_self])
        original_gm = builder.get_graph_module()
        p = ReplaceScalarTensorWithFullPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.where.self),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.scalar_tensor.default),
            0,
        )

    @torch.no_grad()
    def test_replace_scalar_tensor_with_full(
        self,
    ):
        original_gm = single_op_builder(
            placeholders=(),
            op=exir_ops.edge.aten.scalar_tensor.default,
            args=(0.123,),
        )
        p = ReplaceScalarTensorWithFullPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.scalar_tensor.default),
            0,
        )

    @torch.no_grad()
    def test_replace_linear_with_fully_connected(self):
        shape, in_channels, out_channels = (1, 14), 14, 128
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(*shape, dtype=torch.float32))
        weights = builder.placeholder(
            "weights", torch.randn([out_channels, in_channels], dtype=torch.float32)
        )
        permute_copy = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(weights, [1, 0]),
        )
        mm = builder.call_operator(
            op=exir_ops.edge.aten.mm.default,
            args=(x, permute_copy),
        )
        builder.output([mm])
        original_gm = builder.get_graph_module()
        gm = cast(
            PassResult, ReplacePermuteWithTransposePass()(original_gm)
        ).graph_module
        gm = cast(PassResult, ReplaceMMWithAddMMPass()(gm)).graph_module
        gm = cast(PassResult, ReplaceAddMMWithLinearPass()(gm)).graph_module
        graph_after_passes = cast(
            PassResult, ReplaceLinearWithFullyConnectedOpPass()(gm)
        ).graph_module
        self.assertIsNotNone(graph_after_passes)
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default),
            1,
        )
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.fully_connected.default
            ),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear),
            0,
        )

    @parameterized.expand(
        [
            [(4, 16, 256), 256, 512, True],
            [(7, 17, 12), 12, 34, False],
        ]
    )
    @torch.no_grad()
    def test_replace_addmm_with_linear(
        self, shape: Tuple[int], in_features: int, out_features: int, bias: bool
    ):
        M, K, N, alpha, beta = 14, 48, 24, 1.0, 1.0
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(N, dtype=torch.float32))
        y = builder.placeholder("y", torch.randn([M, K], dtype=torch.float32))
        z = builder.placeholder("z", torch.randn([N, K], dtype=torch.float32))
        permute_copy = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(z, [1, 0]),
        )
        addmm = builder.call_operator(
            op=exir_ops.edge.aten.addmm.default,
            args=(x, y, permute_copy),
            kwargs={"beta": beta, "alpha": alpha},
        )
        builder.output([addmm])
        original_gm = builder.get_graph_module()
        gm = cast(
            PassResult, ReplacePermuteWithTransposePass()(original_gm)
        ).graph_module
        graph_after_passes = cast(
            PassResult, ReplaceAddMMWithLinearPass()(gm)
        ).graph_module
        self.assertIsNotNone(graph_after_passes)
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.addmm.default),
            0,
        )

    @torch.no_grad()
    def test_replace_mm_with_addmm(self):
        M, K, N = 14, 48, 24
        x = torch.randn([M, K])
        y = torch.randn([K, N])
        original_gm = single_op_builder(
            placeholders=(x, y),
            op=exir_ops.edge.aten.mm.default,
            args=(x, y),
        )
        p = ReplaceMMWithAddMMPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertIsNotNone(graph_after_passes)
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.addmm.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mm.default),
            0,
        )

    @parameterized.expand(
        [
            # shape
            [(5, 1, 6, 7)],
            [1],
            [(4, 3, 2)],
            # shape, dim to squeeze
            [(2, 1), 0],
            [(2, 7, 1, 3), 1],
            [(2, 1, 3), 2],
        ]
    )
    @torch.no_grad()
    def test_replace_squeeze_with_view(self, shape: Tuple[int], dim=None):
        x = torch.randn(shape)
        if dim:
            original_gm = single_op_builder(
                placeholders=(x,),
                op=exir_ops.edge.aten.squeeze_copy.dim,
                args=(x, dim),
            )
        else:
            original_gm = single_op_builder(
                placeholders=(x,),
                op=exir_ops.edge.aten.squeeze_copy.default,
                args=(x,),
            )
        p = ReplaceSqueezeAndUnsqueezeWithViewPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertIsNotNone(graph_after_passes)
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default),
            1,
        )
        if dim:
            self.assertEqual(
                count_node(graph_after_passes, exir_ops.edge.aten.squeeze_copy.dim),
                0,
            )
        else:
            self.assertEqual(
                count_node(graph_after_passes, exir_ops.edge.aten.squeeze_copy.default),
                0,
            )

    @parameterized.expand(
        [
            # shape, dim to unsqueeze
            [(5, 6, 7), 0],
            [(5, 6, 7), -1],
            [(5, 6, 7), 3],
            [(5, 6, 7), 2],
        ]
    )
    @torch.no_grad()
    def test_replace_unsqueeze_with_view(self, shape: Tuple[int], dim: int):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.unsqueeze_copy.default,
            args=(x, dim),
        )
        p = ReplaceSqueezeAndUnsqueezeWithViewPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertIsNotNone(graph_after_passes)
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.unsqueeze_copy.default),
            0,
        )

    @torch.no_grad()
    def test_replace_single_element_tensor_arguments_from_full_op_with_scalar(
        self,
        in_features: int = 16,
        out_features: int = 16,
    ):
        src_zero_point = 0
        out_zero_point = 0
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn([1, in_features]))
        weights = builder.placeholder(
            "weights", torch.randn([in_features, out_features], dtype=torch.float32)
        )
        bias = builder.placeholder(
            "bias", torch.randn([out_features], dtype=torch.float32)
        )
        quantized_input = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 0.01431146077811718, 57, -128, 127, torch.int8),
        )
        weight_zero_point = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
        )
        out_multiplier = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
        )
        out_shift = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
        )
        output = builder.call_operator(
            op=exir_ops.edge.cadence.quantized_linear.default,
            args=(
                quantized_input,
                weights,
                bias,
                src_zero_point,
                weight_zero_point,
                out_multiplier,
                out_shift,
                out_zero_point,
                None,
            ),
        )
        dequantized_output = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(output, 0.010696045123040676, -31, -128, 127, torch.int8),
        )
        builder.output([dequantized_output])
        original_gm = builder.get_graph_module()
        p = ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertIsNotNone(graph_after_passes)
        gm = dead_code_elimination_pass(graph_after_passes).graph_module
        # By default, the quantized linear op should have constant scalar attributes.
        self.assertTargetCountsEqual(
            gm,
            [
                # No default quantized linear op.
                (exir_ops.edge.cadence.quantized_linear.default, 0),
                # The default quantized linear op will be replaced with quantized_linear.per_tensor.
                (exir_ops.edge.cadence.quantized_linear.per_tensor, 1),
                # No aten.full ops.
                (exir_ops.edge.aten.full.default, 0),
            ],
        )

    @torch.no_grad()
    def test_replace_single_element_tensor_arguments_from_full_op_with_scalar_tuple_args(
        self,
        in_features: int = 16,
        out_features: int = 16,
    ):
        src_zero_point = 0
        out_zero_point = 0
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn([1, in_features]))
        weights = builder.placeholder(
            "weights", torch.randn([in_features, out_features], dtype=torch.float32)
        )
        bias = builder.placeholder(
            "bias", torch.randn([out_features], dtype=torch.float32)
        )
        quantized_input = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(x, 0.01431146077811718, 57, -128, 127, torch.int8),
        )
        weight_zero_point = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
        )
        out_multiplier = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
        )
        out_shift = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1], 0),
        )
        output = builder.call_operator(
            op=exir_ops.edge.cadence.quantized_linear.default,
            args=(
                quantized_input,
                weights,
                bias,
                src_zero_point,
                weight_zero_point,
                out_multiplier,
                out_shift,
                out_zero_point,
                None,
            ),
        )
        dequantized_output = builder.call_operator(
            op=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(output, 0.010696045123040676, -31, -128, 127, torch.int8),
        )
        builder.output([dequantized_output])
        original_gm = builder.get_graph_module()

        for node in original_gm.graph.nodes:
            # Replace the `shape` argument for aten.full op with a tuple.
            if node.target == exir_ops.edge.aten.full.default:
                node.args = (tuple(node.args[0]), node.args[1])

        # Apply replacement pass.
        p = ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertIsNotNone(graph_after_passes)
        gm = dead_code_elimination_pass(graph_after_passes).graph_module

        # By default, the quantized linear op should have constant scalar attributes.
        self.assertTargetCountsEqual(
            gm,
            [
                # No default quantized linear op.
                (exir_ops.edge.cadence.quantized_linear.default, 0),
                # The default quantized linear op will be replaced with quantized_linear.per_tensor.
                (exir_ops.edge.cadence.quantized_linear.per_tensor, 1),
                # No aten.full ops.
                (exir_ops.edge.aten.full.default, 0),
            ],
        )

    @torch.no_grad()
    def test_replace_conv1d_with_linear(self):
        x = torch.randn(1, 96, 7)
        weights = torch.randn(192, 96, 7)
        bias = torch.randn(192)
        original_gm = single_op_builder(
            placeholders=(x, weights, bias),
            op=exir_ops.edge.cadence.convolution.default,
            args=(x, weights, bias, [1], [0], [1], 1, False),
        )
        # First, replace the aten convolution with a cadence.convolution op
        p1 = ReplaceAtenConvolutionWithJarvisConvolutionPass()
        temp_graph = p1(original_gm).graph_module
        self.assertIsNotNone(temp_graph)

        p2 = ReplaceTrivialConvWithLinear()
        graph_after_passes = p2(temp_graph).graph_module

        # Assert that conv1d is trivially converted to linear
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.convolution.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.im2row.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear.default)
            + count_node(
                graph_after_passes, exir_ops.edge.cadence.fully_connected.default
            ),
            1,
        )

    @torch.no_grad()
    def test_replace_conv2d_with_linear(self):
        x = torch.randn(1, 96, 7, 7)
        weights = torch.randn(192, 96, 7, 7)
        bias = torch.randn(192)
        original_gm = single_op_builder(
            placeholders=(x, weights, bias),
            op=exir_ops.edge.cadence.convolution.default,
            args=(x, weights, bias, [1, 1], [0, 0], [1, 1], 1, False),
        )
        # First, replace the aten convolution with a cadence.convolution op
        p1 = ReplaceAtenConvolutionWithJarvisConvolutionPass()
        temp_graph = p1(original_gm).graph_module
        self.assertIsNotNone(temp_graph)

        p2 = ReplaceTrivialConvWithLinear()
        graph_after_passes = p2(temp_graph).graph_module

        # Assert that conv2d is trivially converted to linear
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.convolution.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.im2row.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear.default)
            + count_node(
                graph_after_passes, exir_ops.edge.cadence.fully_connected.default
            ),
            1,
        )

    @torch.no_grad()
    def test_replace_conv2d_with_im2row_and_linear(self):
        x = torch.randn(1, 96, 47, 37)
        weights = torch.randn(192, 96, 7, 7)
        bias = torch.randn(192)
        original_gm = single_op_builder(
            placeholders=(x, weights, bias),
            op=exir_ops.edge.cadence.convolution.default,
            args=(x, weights, bias, [1, 1], [0, 0], [1, 1], 1, False),
        )
        p = ReplaceConvWithIm2RowAndLinear()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module

        # Assert that the convolution is converted to im2row + linear
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.convolution.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.im2row.default), 1
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear.default), 1
        )

    @parameterized.expand(
        [
            [(3, 1, 5), 1, 0],
            [(3, 4, 1), 2, -1],
        ]
    )
    @torch.no_grad()
    def test_replace_select_with_view(self, shape: Tuple[int], dim: int, index: int):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.select_copy.int,
            args=(x, dim, index),
        )
        p = ReplaceSelectWithViewOpPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        # Assert that select op was replaced with view op
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.select_copy.int), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 1
        )

    @parameterized.expand(
        [
            [(2, 1, 3, 1), 1, 3, torch.float32],
            [(2, 1, 5), 1, 0, torch.int64],
            [(3, 1, 5), 0, 1, torch.int64],
        ]
    )
    @torch.no_grad()
    def test_replace_nop_transpose_with_view(
        self,
        shape: Tuple[int],
        dim0: int,
        dim1: int,
        dtype: torch.dtype = torch.float32,
    ):
        if dtype == torch.float32:
            x = torch.randn(shape)
        else:
            x = torch.randint(low=0, high=100, size=shape, dtype=torch.int64)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(x, dim0, dim1),
        )
        p = ReplaceNopTransposeOrPermuteWithViewPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module

        # Assert that transpose op was removed, and a view op was placed instead
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 1
        )

    @parameterized.expand(
        [
            # permutations that can be replaced by view
            [(3, 1, 3, 1, 4), (0, 2, 4, 1, 3)],
            [(1, 3, 4), (1, 2, 0)],
        ]
    )
    @torch.no_grad()
    def test_replace_nop_permute_with_view(self, shape, dims):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, dims),
        )
        p = ReplaceNopTransposeOrPermuteWithViewPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module

        # Assert that permute op was removed, and a view op was placed instead
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 1
        )

    @parameterized.expand(
        [
            # permutations replaced by transpose
            [(3, 4), [1, 0]],
            [(3, 4, 6), (0, 2, 1)],
        ]
    )
    @torch.no_grad()
    def test_replace_permute_with_transpose(self, shape, dims):
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, dims),
        )
        p = ReplacePermuteWithTransposePass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module

        # Assert that permute op was replaced by a transpose op
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int), 1
        )

    @torch.no_grad()
    def test_replace_permute_with_transpose_nop(
        self,
    ):
        x = torch.randn(3, 4)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, [0, 1]),
        )
        p = ReplacePermuteWithTransposePass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module

        # Assert that permute op was replaced by a transpose op
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int), 0
        )

    def test_replace_aten_where_with_cadence(self):
        builder = GraphBuilder()
        cond = builder.placeholder("cond", torch.randn(4, 8))
        aten_gt_scalar = builder.call_operator(
            op=exir_ops.edge.aten.gt.Scalar,
            args=(cond, 0),
        )
        aten_full_default = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([4, 8], 0.0),
        )
        aten_full_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([4, 8], 1.0),
        )
        aten_where_self = builder.call_operator(
            op=exir_ops.edge.aten.where.self,
            args=(aten_gt_scalar, aten_full_default, aten_full_default_1),
        )
        builder.output([aten_where_self])
        original_gm = builder.get_graph_module()
        p = ReplaceWhereWithFullArgsWithWhereScalar()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.where.self,
            ),
            0,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.where_Scalar.default),
            1,
        )

    @parameterized.expand(
        [
            [(4, 8), (4, 8), (4, 8), 0.0, 1.0],
            [(8,), (4, 8), (8,), 0.0, 1.0],
            [(4, 8), (8,), (8,), 0.0, 1.0],
        ]
    )
    def test_replace_aten_where_with_cadence_broadcast(
        self, cond_shape, a_shape, b_shape, val1, val2
    ):
        # cond_shape, a_shape, b_shape, val1, val2 =
        builder = GraphBuilder()
        cond = builder.placeholder("cond", torch.randn(cond_shape))
        aten_gt_scalar = builder.call_operator(
            op=exir_ops.edge.aten.gt.Scalar,
            args=(cond, 0),
        )
        aten_full_default = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(a_shape, val1),
        )
        aten_full_default_1 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(b_shape, val2),
        )
        aten_where_self = builder.call_operator(
            op=exir_ops.edge.aten.where.self,
            args=(aten_gt_scalar, aten_full_default, aten_full_default_1),
        )
        builder.output([aten_where_self])
        original_gm = builder.get_graph_module()
        p = ReplaceWhereWithFullArgsWithWhereScalar()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.where.self,
            ),
            1,
        )

    def test_no_replace_aten_gelu_with_approximate_gelu(self):
        inputs = torch.randn(2, 1, 64)

        gm = single_op_builder(
            placeholders=(inputs,),
            op=exir_ops.edge.aten.gelu.default,
            args=(inputs,),
        )
        gm = ExportPass().call(gm).graph_module

        p = ReplaceAtenApproxGeluWithApproxGeluPass()
        graph_after_passes = p.call(gm).graph_module

        # Assert that aten.gelu op was not decomposed, since it didn't have an approximate argument
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.gelu.default,
            ),
            1,
        )

    def test_replace_split_with_sizes_with_slice(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 16, 8, 4))
        split = builder.call_operator(
            exir_ops.edge.aten.split_with_sizes_copy.default, (x, [8, 8], 1)
        )
        # We need the outputs to be gathered by getitem ops
        out0 = builder.call_operator(operator.getitem, (split, 0))
        out1 = builder.call_operator(operator.getitem, (split, 1))
        builder.output([out0, out1])
        graph_module = builder.get_graph_module()

        p = ReplaceSplitWithSlicePass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.aten.split_with_sizes_copy.default
            ),
            0,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.slice_copy.Tensor),
            2,
        )

    @parameterized.expand([[2], [3], [4]])
    def test_replace_pow_with_mul(self, exponent: int):
        x = torch.randn(2, 1, 64)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.pow.Tensor_Scalar,
            args=(x, exponent),
        )
        p = ReplacePowWithMulPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.pow.Tensor_Scalar,
            ),
            0,
        )
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.mul.Tensor,
            ),
            exponent - 1,
        )

    @parameterized.expand(
        [
            [1],
            [1.5],
        ]
    )
    def test_replace_pow_with_mul_not_applied(self, exponent):
        x = torch.randn(2, 1, 64)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.pow.Tensor_Scalar,
            args=(x, exponent),
        )
        p = ReplacePowWithMulPass()
        graph_after_passes = cast(PassResult, p(original_gm)).graph_module

        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.pow.Tensor_Scalar,
            ),
            1,
        )

        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.mul.Tensor,
            ),
            0,
        )


class TestReplaceIm2rowWithViewPass(unittest.TestCase):
    def test_no_replacement_for_conv(self):
        # Create a graph with a single im2row node.
        x = torch.randn(1, 3, 224, 224)
        pad_value = torch.randn(1)
        channels_last = False
        gm = single_op_builder(
            placeholders=(x, pad_value),
            op=exir_ops.edge.cadence.im2row.default,
            args=(x, (2, 2), (1, 1), (0, 0), (1, 1), pad_value, channels_last),
        )
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.im2row.default), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 0)

        # Apply replacement pass.
        p = ReplaceIm2RowWithViewPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.im2row.default), 1
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.view_copy.default), 0
        )

    def test_no_replace_for_dilation(self):
        # Create a graph with a single im2row node.
        x = torch.randn(1, 3, 5, 7)
        pad_value = torch.randn(1)
        channels_last = False
        gm = single_op_builder(
            placeholders=(x, pad_value),
            op=exir_ops.edge.cadence.im2row.default,
            args=(x, (3, 4), (2, 2), (0, 0), (1, 1), pad_value, channels_last),
        )
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.im2row.default), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 0)

        # Apply replacement pass.
        p = ReplaceIm2RowWithViewPass()
        gm_after_replacement = p.call(gm).graph_module
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.im2row.default), 1
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.view_copy.default), 0
        )

    def test_replace_linear_like_conv(self):
        # Create a graph with a single im2row node.
        in_h, in_w = 13, 15
        x = torch.randn(1, 3, in_h, in_w)
        pad_value = torch.randn(1)
        channels_last = False
        gm = single_op_builder(
            placeholders=(x, pad_value),
            op=exir_ops.edge.cadence.im2row.default,
            args=(x, (in_h, in_w), (1, 1), (0, 0), (1, 1), pad_value, channels_last),
        )
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.im2row.default), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.view_copy.default), 0)

        # Apply replacement pass.
        p = ReplaceIm2RowWithViewPass()
        gm_after_replacement = p.call(gm).graph_module
        # In this test, the kernel width/height is the same as the input width/height.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.im2row.default), 0
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.view_copy.default), 1
        )


class TestForceChannelLastForConvPass(unittest.TestCase):
    def create_conv1d_graphmodule(
        self, channels_last: Optional[bool] = None
    ) -> torch.fx.GraphModule:
        """Helper to create a convolution node.

        convolution(
            Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding,"
            int[] dilation, int groups, bool channel_last=False) -> (Tensor Y)"
        """
        if channels_last:
            x = torch.randn(1, 224, 3)
            w = torch.randn(16, 16, 3)
        else:
            x = torch.randn(1, 3, 224)
            w = torch.randn(16, 3, 16)
        b = torch.randn(16)
        args = (x, w, b, (2, 2), (1, 1), (0, 0), 1)
        if channels_last is not None:
            args = args + (channels_last,)
        return single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.convolution.default,
            args=args,
        )

    def test_conv1d_default_channel_last(self):
        # Create a graph with a single convolution node.
        # Check if graph module is valid by running exportpass on it.
        gm = self.create_conv1d_graphmodule()
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.convolution.default), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.transpose_copy.int), 0)

        # Apply replacement pass.
        p = ForceChannelLastForConvPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.convolution.default),
            1,
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.transpose_copy.int),
            # Two transposes are added, one for the input and one for the output.
            3,
        )
        for node in gm_after_replacement.graph.nodes:
            if node.target != exir_ops.edge.cadence.convolution.default:
                continue
            # Check that the channel_last argument is set to True.
            self.assertEqual(len(node.args), 8, f"{node=}")
            self.assertTrue(node.args[7])

    def test_conv1d_no_transpose_if_already_channel_last(self):
        gm = self.create_conv1d_graphmodule(channels_last=True)
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.convolution.default), 1)

        # Apply replacement pass.
        p = ForceChannelLastForConvPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.convolution.default),
            1,
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.transpose_copy.int),
            0,
        )
        for node in gm_after_replacement.graph.nodes:
            if node.target != exir_ops.edge.cadence.convolution.default:
                continue
            # Check that the channel_last argument is set to True.
            self.assertEqual(len(node.args), 8, f"{node=}")
            self.assertTrue(node.args[7])

    def create_convolution_graph_module(
        self, channels_last: Optional[bool] = None
    ) -> torch.fx.GraphModule:
        """Helper to create a convolution node.

        convolution(
            Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding,"
            int[] dilation, int groups, bool channel_last=False) -> (Tensor Y)"
        """
        if channels_last:
            x = torch.randn(1, 224, 224, 3)
            w = torch.randn(16, 16, 16, 3)
        else:
            x = torch.randn(1, 3, 224, 224)
            w = torch.randn(16, 3, 16, 16)
        b = torch.randn(16)
        args = (x, w, b, (2, 2), (1, 1), (0, 0), 1)
        if channels_last is not None:
            args = args + (channels_last,)
        return single_op_builder(
            placeholders=(x, w, b),
            op=exir_ops.edge.cadence.convolution.default,
            args=args,
        )

    def test_convolution_default_channel_last(self):
        # Create a graph with a single convolution node.
        # Check if graph module is valid by running exportpass on it.
        gm = self.create_convolution_graph_module()
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.convolution.default), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)

        # Apply replacement pass.
        p = ForceChannelLastForConvPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.convolution.default),
            1,
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.permute_copy.default),
            # Three permutes are added, two for the input/weights and one for the output.
            3,
        )
        for node in gm_after_replacement.graph.nodes:
            if node.target != exir_ops.edge.cadence.convolution.default:
                continue
            # Check that the channel_last argument is set to True.
            self.assertEqual(len(node.args), 8, f"{node=}")
            self.assertTrue(node.args[7])

    def test_no_transpose_if_already_channel_last(self):
        gm = self.create_convolution_graph_module(channels_last=True)
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.convolution.default), 1)

        # Apply replacement pass.
        p = ForceChannelLastForConvPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.convolution.default),
            1,
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.permute_copy.default),
            0,
        )
        for node in gm_after_replacement.graph.nodes:
            if node.target != exir_ops.edge.cadence.convolution.default:
                continue
            # Check that the channel_last argument is set to True.
            self.assertEqual(len(node.args), 8, f"{node=}")
            self.assertTrue(node.args[7])

    def create_quantized_convolution_graph_module(
        self, channels_last: Optional[bool] = None
    ) -> torch.fx.GraphModule:
        """Helper to create a quantized conv node.

        quantized_conv(
            Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding,
            int[] dilation, int groups, int input_zero_point, Tensor weight_zero_point,
            Tensor bias_scale, float out_scale, int out_zero_point, Tensor out_multiplier,
            Tensor out_shift, bool channel_last=False) -> (Tensor Z)"
        """
        if channels_last:
            x = torch.randn(1, 224, 56, 3)
            w = torch.randn(16, 16, 16, 3)
        else:
            x = torch.randn(1, 3, 224, 56)
            w = torch.randn(16, 3, 16, 16)
        b = torch.randn(16)
        stride = (2, 2)
        padding = (0, 0)
        dilation = (1, 1)
        groups = 1
        input_zero_point = 0
        w_zero_point = torch.randn(1)
        b_scale = torch.randn(1)
        out_scale = 1
        out_zero_point = 0
        out_multiplier = torch.randn(1)
        out_shift = torch.randn(1)
        args = (
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
            input_zero_point,
            w_zero_point,
            b_scale,
            out_scale,
            out_zero_point,
            out_multiplier,
            out_shift,
        )
        if channels_last is not None:
            args = args + (channels_last,)
        return single_op_builder(
            placeholders=(x, w, b, w_zero_point, b_scale, out_multiplier, out_shift),
            op=exir_ops.edge.cadence.quantized_conv.default,
            args=args,
        )

    def test_quantized_convolution_default_channel_last(self):
        # Create a graph with a single convolution node.
        gm = self.create_quantized_convolution_graph_module()
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv.default), 1
        )
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)

        # Apply replacement pass.
        p = ForceChannelLastForConvPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(
                gm_after_replacement, exir_ops.edge.cadence.quantized_conv.default
            ),
            1,
        )
        # Three permutes are added, two for the input/weights and one for the output.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.permute_copy.default),
            3,
        )
        for node in gm_after_replacement.graph.nodes:
            if node.target != exir_ops.edge.cadence.quantized_conv.default:
                continue
            # Check that the channel_last argument is set to True.
            self.assertEqual(len(node.args), 15, f"{node=}")
            self.assertTrue(node.args[14])

    def test_no_transpose_if_already_quantized_conv_channel_last(self):
        # Create a graph with a single im2row node.
        gm = self.create_quantized_convolution_graph_module(channels_last=True)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv.default), 1
        )

        # Apply replacement pass.
        p = ForceChannelLastForConvPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(
                gm_after_replacement, exir_ops.edge.cadence.quantized_conv.default
            ),
            1,
        )
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)
        for node in gm_after_replacement.graph.nodes:
            if node.target != exir_ops.edge.cadence.quantized_conv.default:
                continue
            # Check that the channel_last argument is set to True.
            self.assertEqual(len(node.args), 15, f"{node=}")
            self.assertTrue(node.args[14])


class TestMakeSliceAndCatDimOutermostPass(unittest.TestCase):
    def create_slice_graph(
        self,
        input_shape: Sequence[int],
        slice_dim: int,
        slice_begin: Optional[int] = None,
        slice_end: Optional[int] = None,
    ) -> torch.fx.GraphModule:
        x = torch.randn(*input_shape)
        return single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(x, slice_dim, slice_begin, slice_end),
        )

    def test_slice_no_transpose_if_already_outermost(self):
        # Create a graph with a single slice node.
        gm = self.create_slice_graph((3, 224, 224), 0, 1, 2)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_copy.Tensor), 1)

        # Apply replacement pass.
        gm_after_pass = MakeSliceAndCatDimOutermostPass()(gm).graph_module

        # Assert that no transpose ops were added.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_slice_no_transpose_if_outermost_dimensions_are_one(self):
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_slice_graph((1, 3, 4, 6), 1, 1, 2)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_copy.Tensor), 1)

        # Apply replacement pass.
        gm_after_pass = MakeSliceAndCatDimOutermostPass()(gm).graph_module

        # Assert that no transpose ops were added. The slice is on the second
        # outermost dimension, but the outermost dimension is already 1.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_slice_insert_transpose(self):
        # Create a graph with a single slice node.
        gm = self.create_slice_graph((1, 3, 4, 6), 2, 1, 2)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_copy.Tensor), 1)

        # Apply replacement pass.
        gm_after_pass = MakeSliceAndCatDimOutermostPass()(gm).graph_module

        # Assert that there are two transpose ops added.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            2,
        )

    def create_cat_graph(
        self,
        input_shapes: Sequence[Sequence[int]],
        cat_dim: int = 0,
    ) -> torch.fx.GraphModule:
        input_tensors = tuple(torch.randn(s) for s in input_shapes)
        return single_op_builder(
            placeholders=input_tensors,
            op=exir_ops.edge.aten.cat.default,
            args=(input_tensors, cat_dim),
        )

    def test_cat_no_transpose_if_already_outermost(self):
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_cat_graph(input_shapes=((1, 3, 5), (2, 3, 5)), cat_dim=0)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.cat.default), 1)

        # Apply replacement pass.
        gm_after_pass = MakeSliceAndCatDimOutermostPass()(gm).graph_module

        # Assert that no transpose ops were added. The slice is on the second
        # outermost dimension, but the outermost dimension is already 1.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_cat_no_transpose_if_outermost_dimensions_are_one(self):
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_cat_graph(input_shapes=((1, 1, 3, 5), (1, 2, 3, 5)), cat_dim=1)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.cat.default), 1)

        # Apply replacement pass.
        gm_after_pass = MakeSliceAndCatDimOutermostPass()(gm).graph_module

        # Assert that no transpose ops were added. The slice is on the second
        # outermost dimension, but the outermost dimension is already 1.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_cat_insert_transpose(self):
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_cat_graph(
            input_shapes=((1, 1, 3, 5), (1, 1, 3, 3)), cat_dim=-1
        )
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.cat.default), 1)

        # Apply replacement pass.
        gm_after_pass = MakeSliceAndCatDimOutermostPass()(gm).graph_module

        # Assert that transpose ops were added to make cat on outermost dimension.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            3,
        )


class TestReplaceEmptyTensorsWithFullPass(unittest.TestCase):
    def _get_slice_empty_gm(self) -> torch.fx.GraphModule:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(4))
        # This is empty (numel == 0).
        slice0 = builder.call_operator(
            exir_ops.edge.aten.slice_copy.Tensor, (x, 0, 0, 0)
        )
        # Copy of x.
        slice1 = builder.call_operator(exir_ops.edge.aten.slice_copy.Tensor, (x,))
        cat = builder.call_operator(
            exir_ops.edge.aten.cat.default,
            ((slice0, slice1),),
        )
        builder.output([cat])
        return builder.get_graph_module()

    def test_empty_slice(self):
        gm = self._get_slice_empty_gm()
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=exir_ops.edge.aten.slice_copy.Tensor
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=exir_ops.edge.aten.full.default
                )
            ),
            0,
        )
        updated_gm = ReplaceEmptyTensorsWithFullPass()(gm).graph_module
        self.assertEqual(
            len(
                updated_gm.graph.find_nodes(
                    op="call_function", target=exir_ops.edge.aten.slice_copy.Tensor
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                updated_gm.graph.find_nodes(
                    op="call_function", target=exir_ops.edge.aten.full.default
                )
            ),
            1,
        )
