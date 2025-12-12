# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import operator
import unittest
from typing import cast, List, Optional, Sequence, Tuple, Union

import executorch.backends.cadence.aot.ref_implementations  # noqa

import torch
from executorch.backends.cadence.aot.graph_builder import (
    GraphBuilder,
    single_op_builder,
)
from executorch.backends.cadence.aot.pass_utils import count_node, op_counts_match
from executorch.backends.cadence.aot.replace_ops import (
    MakeSliceAndCatDimOutermostPass,
    ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass,
    ReplaceAddMMWithLinearPass,
    ReplaceAtenAvgPoolWithCadenceAvgPoolPass,
    ReplaceAtenConvolutionWithCadenceConvolutionPass,
    ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass,
    ReplaceConstantPadNdWithSlicePass,
    ReplaceConvolutionOptionalArgsWithConcreteArgsPass,
    ReplaceConvWithChannelLastConvPass,
    ReplaceConvWithIm2RowAndLinear,
    ReplaceEmptyTensorsWithFullPass,
    ReplaceFunctionallyEquivalentOpTargets,
    ReplaceIm2RowWithViewPass,
    ReplaceLinearWithFullyConnectedOpPass,
    ReplaceLogicalNotBooleanWhereWithWherePass,
    ReplaceMatmulWithTransposedMatmulPass,
    ReplaceMMWithAddMMPass,
    ReplaceMulTensorWithMulAndFullOpsPass,
    ReplaceNopTransposeOrPermuteWithViewPass,
    ReplacePadWithCatPass,
    ReplacePermuteWithTransposePass,
    ReplacePowWithMulPass,
    ReplaceRepeatWithCatPass,
    ReplaceScalarTensorWithFullPass,
    ReplaceScalarWithTensorArgPass,
    ReplaceSelectWithViewOpPass,
    ReplaceSplitWithSlicePass,
    ReplaceSqueezeAndUnsqueezeWithViewPass,
    ReplaceTorchQuantizedEmbeddingWithCadenceQuantizedEmbedding,
    ReplaceTransposedConvWithLinearPass,
    ReplaceTrivialConvWithLinear,
    ReplaceWhereWithFullArgsWithWhereScalar,
)

from executorch.backends.cadence.aot.typing_stubs import expand
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, ProxyValue
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils import _pytree as pytree


def validate(
    original: torch.fx.GraphModule,
    modified: torch.fx.GraphModule,
    inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    pass_name: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Validate that two graph modules produce numerically equivalent outputs.

    Args:
        original: The original graph module before the pass
        modified: The modified graph module after the pass
        inputs: Input tensors to run through both graphs
        pass_name: Name of the pass being validated (for error messages)
        rtol: Relative tolerance for allclose comparison
        atol: Absolute tolerance for allclose comparison
    """
    original.eval()
    modified.eval()
    with torch.no_grad():
        orig_out = original(*inputs)
        mod_out = modified(*inputs)

    flat_orig_out, _ = pytree.tree_flatten(orig_out)
    flat_mod_out, _ = pytree.tree_flatten(mod_out)

    # Check that outputs match within tolerance
    for i, (orig_tensor, mod_tensor) in enumerate(zip(flat_orig_out, flat_mod_out)):
        if not torch.allclose(orig_tensor, mod_tensor, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(orig_tensor - mod_tensor)).item()
            raise AssertionError(
                f"Pass validation failed for pass {pass_name}. "
                f"Output tensor {i} differs by max {max_diff:.6e}. "
                f"Expected rtol={rtol}, atol={atol}. "
                f"Original output: {orig_tensor}, Modified output: {mod_tensor}"
            )


class TestReplaceOpsPasses(unittest.TestCase):
    def assertTargetCountEqual(
        self,
        graph_module: torch.fx.GraphModule,
        target: torch.fx.node.Target,
        expected_count: int,
    ) -> None:
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
        targets_and_counts: List[Tuple[torch.fx.node.Target, int]],
    ) -> None:
        """Helper function to check the number of nodes of all types for a given target."""
        for target, expected_count in targets_and_counts:
            self.assertTargetCountEqual(graph_module, target, expected_count)

    @expand(
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
        _: str,
        x_shape: Tuple[int],
        y_shape: Tuple[int],
    ) -> None:
        builder = GraphBuilder()
        x_ = torch.randint(0, 100, x_shape, dtype=torch.int8)
        x = builder.placeholder("x", x_)
        y_ = torch.randint(0, 100, y_shape, dtype=torch.int8)
        y = builder.placeholder("y", y_)
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

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceMatmulWithTransposedMatmulPass()
        result = p.call(original_gm)
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

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
        validate(
            gm_before,
            graph_after_passes,
            (x_, y_),
            "ReplaceMatmulWithTransposedMatmulPass",
        )

    @expand(
        [
            ("2d", (3, 5), [0, 0]),  # shape  # padding
            ("3d", (20, 1, 80), [0, 0, 0]),  # shape  # padding
        ],
    )
    @torch.no_grad()
    def test_replace_constant_pad_nd_with_slice(
        self, _, shape: Tuple[int], padding: Tuple[int]
    ) -> None:
        builder = GraphBuilder()
        x_input = torch.randn(*shape, dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        matmul = builder.call_operator(
            op=exir_ops.edge.aten.constant_pad_nd.default,
            args=(x, [0, 0, 0, 0]),
        )
        builder.output([matmul])
        original_gm = builder.get_graph_module()

        # Deepcopy before the pass
        gm_before = copy.deepcopy(original_gm)
        p = ReplaceConstantPadNdWithSlicePass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate(
            gm_before, graph_after_passes, inputs, "ReplaceConstantPadNdWithSlicePass"
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.slice.Tensor),
            1,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.constant_pad_nd.default),
            0,
        )

    @expand(
        [
            ["3d", (7, 5, 6), 1.23],
            ["2d", (7, 5), 2],
            ["1d", (10,), 42949],
        ]
    )
    @torch.no_grad()
    def test_add_replace_scalar_with_tensor_arg(
        self, _, shape: Tuple[int], other: float
    ) -> None:
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

    @expand(
        [
            ["3d", (7, 5, 6), 1.23],
            ["2d", (7, 5), 2],
            ["1d", (10,), 42949],
        ]
    )
    @torch.no_grad()
    def test_sub_replace_scalar_with_tensor_arg(
        self, _, shape: Tuple[int], other: float
    ) -> None:
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

    @expand(
        [
            ["3d", (7, 5, 6), 1.23],
            ["2d", (7, 5), 2],
            ["1d", (10,), 42949],
        ]
    )
    @torch.no_grad()
    def test_mul_replace_scalar_with_tensor_arg(
        self, _, shape: Tuple[int], other: float
    ) -> None:
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

    @expand(
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
    ) -> None:
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

    @expand(
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
    ) -> None:
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.relu_.default,
            args=(x,),
        )
        p = ReplaceFunctionallyEquivalentOpTargets()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.relu.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.relu_.default),
            0,
        )

    @expand(
        [
            ("split_linear_tensor_split_size_2", (50,), 2, 0),
            ("split_linear_tensor_split_size_5", (50,), 5, 0),
            ("split_linear_tensor_split_size_7", (50,), 7, 0),
            ("split_leading_dim_split_size_2", (10, 2, 3), 2, 0),
            ("split_leading_dim_split_size_5", (10, 2, 3), 5, 0),
            ("split_leading_dim_split_size_7", (10, 2, 3), 7, 0),
            ("split_trailing_dim_split_size_2", (3, 3, 6), 2, 2),
            ("split_trailing_dim_split_size_4", (3, 3, 6), 4, 2),
            ("split_trailing_dim_split_size_6", (3, 3, 6), 6, 2),
            ("split_middle_dim_split_size_2", (3, 5, 14, 2, 3), 2, 2),
            ("split_middle_dim_split_size_5", (3, 5, 14, 2, 3), 5, 2),
            ("split_middle_dim_split_size_7", (3, 5, 14, 2, 3), 7, 2),
        ]
    )
    @torch.no_grad()
    def test_replace_functionally_equivalent_op_targets_unsafe_split(
        self, _, shape: Tuple[int], split_size: int, dim: int
    ) -> None:
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

    def assertTensorMetadataIsSame(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> None:
        for i, (_a, _b) in enumerate(zip(a, b)):
            # TODO: actually compare the tensors.
            self.assertTrue(
                _a.shape == _b.shape, f"Tensor {i}: {_a.shape} != {_b.shape}"
            )
            self.assertTrue(
                _a.dtype == _b.dtype, f"Tensor {i}: {_a.dtype} != {_b.dtype}"
            )

    @expand(
        [
            [(1, 8, 18), 8, 16, 3],
            [(1, 8, 18), 8, 16, 5, 2],
            # depthwise + bias
            [(1, 8, 18), 8, 16, 5, 2, 0, 1, True],
            # no bias
            [(1, 8, 18), 8, 16, 3, 2, 4, 3, False, False],
            # bias + transposed
            [(1, 8, 18), 8, 16, 5, 2, 0, 1, False, True],
            # Stride of 2 needed.
            [(1, 8, 3), 8, 8, 48, 2, 23],
        ]
    )
    @torch.no_grad()
    def test_replace_aten_conv_with_cadence_conv(
        self,
        shape: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        depthwise: bool = False,
        bias_enabled: bool = True,
        output_padding: Optional[int] = None,
    ) -> None:
        groups = in_channels if depthwise else 1
        builder = GraphBuilder()
        x_tensor = torch.randn(*shape, dtype=torch.float32)
        x = builder.placeholder("x", x_tensor)
        # For regular conv: weight shape is [out_channels, in_channels // groups, kernel]
        weights_shape = [out_channels, in_channels // groups, kernel]
        weights_tensor = torch.randn(weights_shape, dtype=torch.float32)
        weights = builder.placeholder("weights", weights_tensor)
        bias: Optional[ProxyValue] = None
        bias_tensor: Optional[torch.Tensor] = None
        if bias_enabled:
            bias_tensor = torch.randn([out_channels], dtype=torch.float32)
            bias = builder.placeholder("bias", bias_tensor)
        convolution = builder.call_operator(
            op=exir_ops.edge.aten.convolution.default,
            args=(
                x,
                weights,
                bias,
                [stride],
                [padding],
                [dilation],
                False,
                [output_padding] if output_padding else [0],
                groups,
            ),
        )
        builder.output([convolution])
        original_gm = builder.get_graph_module()

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceAtenConvolutionWithCadenceConvolutionPass()
        replacement_pass_result = cast(PassResult, p(original_gm))
        self.assertIsNotNone(replacement_pass_result)
        self.assertTrue(replacement_pass_result.modified)
        graph_after_passes = replacement_pass_result.graph_module

        # Validate numerical accuracy
        inputs = (x_tensor, weights_tensor)
        if bias is not None:
            inputs += (cast(torch.Tensor, bias_tensor),)
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceAtenConvolutionWithCadenceConvolutionPass",
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.convolution.default),
            0,
        )
        # This is a 1D convolution (using [stride], [padding], [dilation])
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.conv1d.default),
            1,
        )
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.transposed_convolution.default
            ),
            0,
        )

    @expand(
        [
            [(1, 8, 16), 8, 16, 3],
            [(1, 8, 16), 8, 16, 5, 2],
            # depthwise + bias
            [(1, 8, 16), 8, 16, 5, 2, 0, 1, True, True],
            # no bias
            [(1, 8, 16), 8, 16, 3, 2, 4, 3, False, False],
            # depthwise + no bias
            [(1, 8, 16), 8, 16, 3, 1, 0, 1, True, False],
            # bias
            [(1, 8, 16), 8, 16, 5, 2, 0, 1, False, True],
        ]
    )
    @torch.no_grad()
    def test_replace_aten_transposed_conv_with_cadence_transposed_conv(
        self,
        shape: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        depthwise: bool = False,
        bias_enabled: bool = True,
        output_padding: Optional[int] = None,
    ) -> None:
        groups = in_channels if depthwise else 1
        builder = GraphBuilder()
        x_tensor = torch.randn(*shape, dtype=torch.float32)
        x = builder.placeholder("x", x_tensor)
        # For transposed conv: weight shape is [in_channels, out_channels // groups, kernel]
        weights_shape = [in_channels, out_channels // groups, kernel]
        weights_tensor = torch.randn(weights_shape, dtype=torch.float32)
        weights = builder.placeholder(
            "weights",
            weights_tensor,
        )
        bias_tensor = (
            torch.randn([out_channels], dtype=torch.float32) if bias_enabled else None
        )
        bias = (
            builder.placeholder("bias", cast(torch.Tensor, bias_tensor))
            if bias_enabled
            else None
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
                True,
                [output_padding] if output_padding else [0],
                groups,
            ),
        )
        builder.output([convolution])
        original_gm = builder.get_graph_module()
        gm_before = copy.deepcopy(original_gm)

        p = ReplaceAtenConvolutionWithCadenceConvolutionPass()
        replacement_pass_result = cast(PassResult, p(original_gm))
        self.assertIsNotNone(replacement_pass_result)
        self.assertTrue(replacement_pass_result.modified)
        graph_after_passes = replacement_pass_result.graph_module

        inputs = (x_tensor, weights_tensor)
        if bias_tensor is not None:
            inputs += (bias_tensor,)

        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceAtenConvolutionWithCadenceConvolutionPass",
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.convolution.default),
            0,
        )
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.transposed_convolution.default
            ),
            1,
        )

    @expand(
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
        shape: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        depthwise: bool = False,
        bias_enabled: bool = True,
        channel_last: bool = False,
    ) -> None:
        output_padding = [0]
        groups = in_channels if depthwise else 1
        builder = GraphBuilder()
        x_tensor = torch.randn(*shape, dtype=torch.float32)
        x = builder.placeholder("x", x_tensor)
        # For transposed conv: weight shape is [in_channels, out_channels // groups, kernel]
        weights_tensor = torch.randn(
            [in_channels, out_channels // groups, kernel], dtype=torch.float32
        )
        weights = builder.placeholder("weights", weights_tensor)

        transposed_weights = builder.call_operator(
            op=exir_ops.edge.aten.transpose_copy.int, args=(weights, 0, 1)
        )
        flipped_weights = builder.call_operator(
            exir_ops.edge.aten.flip.default,
            args=(transposed_weights, [-1]),
        )
        bias_tensor = (
            torch.randn([out_channels], dtype=torch.float32) if bias_enabled else None
        )
        bias = (
            builder.placeholder("bias", cast(torch.Tensor, bias_tensor))
            if bias_enabled
            else None
        )
        if channel_last:
            x = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(x, [0, 2, 1]),
            )
        convolution = builder.call_operator(
            op=exir_ops.edge.cadence.transposed_convolution.default,
            args=(
                x,
                flipped_weights,
                bias,
                [stride],
                [padding],
                [dilation],
                output_padding,
                groups,
                False,
            ),
        )
        if channel_last:
            convolution = builder.call_operator(
                op=exir_ops.edge.aten.permute_copy.default,
                args=(convolution, [0, 2, 1]),
            )
        builder.output([convolution])
        original_gm = builder.get_graph_module()

        gm_before = copy.deepcopy(original_gm)

        # Run ReplaceTransposedConvWithLinearPass
        result = ReplaceTransposedConvWithLinearPass().call(original_gm)
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = (x_tensor, weights_tensor)
        if bias_tensor is not None:
            inputs += (bias_tensor,)
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceTransposedConvWithLinearPass",
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.convolution.default),
            0,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.conv1d.default)
            + count_node(graph_after_passes, exir_ops.edge.cadence.conv2d.default),
            0,
        )
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.transposed_convolution.default
            ),
            0,
        )

    @expand(
        [
            [(1, 8, 33), 8, 16, 3, 2, 4, 3, False, False],
            # # depthwise
            [(1, 8, 33), 8, 16, 3, 1, 0, 1, True, False],
            [(1, 8, 33), 8, 16, 3, 2, 4, 3, True, False],
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
    ) -> None:
        groups = in_channels if depthwise else 1
        builder = GraphBuilder()
        x_input = torch.randn(*shape, dtype=torch.float32)
        weights_input = torch.randn(
            [out_channels, in_channels // groups, kernel], dtype=torch.float32
        )
        x = builder.placeholder("x", x_input)
        weights = builder.placeholder("weights", weights_input)
        bias_input = None
        if bias_enabled:
            bias_input = torch.randn([out_channels], dtype=torch.float32)
            bias = builder.placeholder("bias", bias_input)
        else:
            bias = None

        convolution = builder.call_operator(
            op=exir_ops.edge.cadence.conv1d.default,
            args=(
                x,
                weights,
                bias,
                [stride],
                [padding],
                [dilation],
                groups,
            ),
        )
        builder.output([convolution])
        original_gm = builder.get_graph_module()

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceConvolutionOptionalArgsWithConcreteArgsPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        inputs = [x_input, weights_input] + (
            [bias_input] if bias_input is not None else []
        )
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceConvolutionOptionalArgsWithConcreteArgsPass",
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.conv1d.default),
            1,
        )

    @expand(
        [
            [(1, 2, 3), [1, 1]],
            [
                (20, 1, 80),
                [1, 4],
            ],
        ]
    )
    @torch.no_grad()
    def test_replace_pad_with_cat(self, shape: Tuple[int], padding: Tuple[int]) -> None:
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.constant_pad_nd.default,
            args=(x, padding),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplacePadWithCatPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x]
        validate(gm_before, graph_after_passes, inputs, "ReplacePadWithCatPass")

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.constant_pad_nd.default),
            0,
        )

    @torch.no_grad()
    def test_replace_repeat_with_cat(self) -> None:
        x = torch.randn([3, 5])
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.repeat.default,
            args=(x, [1, 2]),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceRepeatWithCatPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        inputs = [x]
        validate(gm_before, graph_after_passes, inputs, "ReplaceRepeatWithCatPass")

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.repeat.default),
            0,
        )

    @expand(
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
    ) -> None:
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
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module
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
    ) -> None:
        original_gm = single_op_builder(
            placeholders=(),
            op=exir_ops.edge.aten.scalar_tensor.default,
            args=(0.123,),
        )
        p = ReplaceScalarTensorWithFullPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.scalar_tensor.default),
            0,
        )

    @torch.no_grad()
    def test_replace_linear_with_fully_connected(self) -> None:
        shape, in_channels, out_channels = (1, 14), 14, 128
        builder = GraphBuilder()
        x_input = torch.randn(*shape, dtype=torch.float32)
        weights_input = torch.randn([out_channels, in_channels], dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        weights = builder.placeholder("weights", weights_input)
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

        gm_before_linear = copy.deepcopy(gm)
        pass_result = cast(PassResult, ReplaceAddMMWithLinearPass()(gm))
        self.assertTrue(pass_result.modified)
        gm = pass_result.graph_module

        inputs = [x_input, weights_input]
        validate(gm_before_linear, gm, inputs, "ReplaceAddMMWithLinearPass")
        gm_before_fc = copy.deepcopy(gm)
        graph_after_passes = cast(
            PassResult, ReplaceLinearWithFullyConnectedOpPass()(gm)
        ).graph_module

        validate(
            gm_before_fc,
            graph_after_passes,
            inputs,
            "ReplaceLinearWithFullyConnectedOpPass",
        )

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

    @expand([[1.0, 1.0], [2.0, 3.0]])
    @torch.no_grad()
    def test_replace_addmm_with_linear(self, alpha: float, beta: float) -> None:
        M, K, N = 14, 12, 10
        builder = GraphBuilder()
        x_input = torch.randn(N, dtype=torch.float32)
        y_input = torch.randn([M, K], dtype=torch.float32)
        z_input = torch.randn([N, K], dtype=torch.float32)
        x = builder.placeholder("x", x_input)
        y = builder.placeholder("y", y_input)
        z = builder.placeholder("z", z_input)
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

        gm_before_linear = copy.deepcopy(gm)
        pass_result = cast(PassResult, ReplaceAddMMWithLinearPass()(gm))
        self.assertTrue(pass_result.modified)
        graph_after_passes = pass_result.graph_module

        inputs = [x_input, y_input, z_input]
        validate(
            gm_before_linear, graph_after_passes, inputs, "ReplaceAddMMWithLinearPass"
        )

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
    def test_replace_mm_with_addmm(self) -> None:
        M, K, N = 14, 48, 24
        x = torch.randn([M, K])
        y = torch.randn([K, N])
        original_gm = single_op_builder(
            placeholders=(x, y),
            op=exir_ops.edge.aten.mm.default,
            args=(x, y),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceMMWithAddMMPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x, y]
        validate(gm_before, graph_after_passes, inputs, "ReplaceMMWithAddMMPass")

        self.assertIsNotNone(graph_after_passes)
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.addmm.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mm.default),
            0,
        )

    @expand(
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
    def test_replace_squeeze_with_view(
        self, shape: Tuple[int], dim: Optional[int] = None
    ) -> None:
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
        result = cast(PassResult, p(original_gm))

        # Assert: Verify the pass modified the graph
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

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

    @expand(
        [
            # shape, dim to unsqueeze
            [(5, 6, 7), 0],
            [(5, 6, 7), -1],
            [(5, 6, 7), 3],
            [(5, 6, 7), 2],
        ]
    )
    @torch.no_grad()
    def test_replace_unsqueeze_with_view(self, shape: Tuple[int], dim: int) -> None:
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.unsqueeze_copy.default,
            args=(x, dim),
        )
        p = ReplaceSqueezeAndUnsqueezeWithViewPass()
        result = cast(PassResult, p(original_gm))

        # Assert: Verify the pass modified the graph
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

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
    def test_replace_squeeze_and_unsqueeze_with_view_no_modification(self) -> None:
        """Negative test: pass doesn't modify graphs without squeeze/unsqueeze ops."""
        x = torch.randn(2, 3, 4)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [2, 12]),
        )
        p = ReplaceSqueezeAndUnsqueezeWithViewPass()
        result = cast(PassResult, p(original_gm))

        # Assert: Verify the pass did NOT modify the graph
        self.assertFalse(result.modified)
        graph_after_passes = result.graph_module

        # Verify the original view_copy operation is still there
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default),
            1,
        )

    @torch.no_grad()
    def test_replace_conv1d_with_linear(self) -> None:
        x = torch.randn(1, 96, 7)
        weights = torch.randn(192, 96, 7)
        bias = torch.randn(192)
        original_gm = single_op_builder(
            placeholders=(x, weights, bias),
            op=exir_ops.edge.cadence.conv1d.default,
            args=(x, weights, bias, [1], [0], [1], 1),
        )

        gm_before = copy.deepcopy(original_gm)
        p2 = ReplaceTrivialConvWithLinear()
        result = cast(PassResult, p2(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x, weights, bias]
        validate(gm_before, graph_after_passes, inputs, "ReplaceTrivialConvWithLinear")

        # Assert that conv1d is trivially converted to linear
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.conv1d.default), 0
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
    def test_replace_conv2d_with_linear(self) -> None:
        x = torch.randn(1, 6, 7, 7)
        weights = torch.randn(12, 6, 7, 7)
        bias = torch.randn(12)
        original_gm = single_op_builder(
            placeholders=(x, weights, bias),
            op=exir_ops.edge.cadence.conv2d.default,
            args=(x, weights, bias, [1, 1], [0, 0], [1, 1], 1),
        )

        gm_before = copy.deepcopy(original_gm)
        p2 = ReplaceTrivialConvWithLinear()
        result = cast(PassResult, p2(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x, weights, bias]
        validate(gm_before, graph_after_passes, inputs, "ReplaceTrivialConvWithLinear")

        # Assert that conv2d is trivially converted to linear
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.conv2d.default), 0
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
    def test_replace_conv2d_with_im2row_and_linear(self) -> None:
        x = torch.randn(1, 2, 5, 5)
        weights = torch.randn(3, 2, 4, 4)
        bias = torch.randn(3)
        original_gm = single_op_builder(
            placeholders=(x, weights, bias),
            op=exir_ops.edge.cadence.conv2d.default,
            args=(x, weights, bias, [1, 1], [0, 0], [1, 1], 1),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceConvWithIm2RowAndLinear()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x, weights, bias]
        validate(
            gm_before, graph_after_passes, inputs, "ReplaceConvWithIm2RowAndLinear"
        )

        # Assert that the convolution is converted to im2row + linear
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.conv2d.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.im2row.per_tensor), 1
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.linear.default), 1
        )

    @expand(
        [
            [(3, 1, 5), 1, 0],
            [(3, 4, 1), 2, -1],
        ]
    )
    @torch.no_grad()
    def test_replace_select_with_view(
        self, shape: Tuple[int], dim: int, index: int
    ) -> None:
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.select_copy.int,
            args=(x, dim, index),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceSelectWithViewOpPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x]
        validate(gm_before, graph_after_passes, inputs, "ReplaceSelectWithViewOpPass")

        # Assert that select op was replaced with view op
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.select_copy.int), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 1
        )

    @expand(
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
    ) -> None:
        if dtype == torch.float32:
            x = torch.randn(shape)
        else:
            x = torch.randint(low=0, high=100, size=shape, dtype=torch.int64)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.transpose_copy.int,
            args=(x, dim0, dim1),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceNopTransposeOrPermuteWithViewPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x]
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceNopTransposeOrPermuteWithViewPass",
        )

        # Assert that transpose op was removed, and a view op was placed instead
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 1
        )

    @expand(
        [
            # permutations that can be replaced by view
            [(3, 1, 3, 1, 4), (0, 2, 4, 1, 3)],
            [(1, 3, 4), (1, 2, 0)],
        ]
    )
    @torch.no_grad()
    def test_replace_nop_permute_with_view(
        self, shape: Tuple[int], dims: Tuple[int]
    ) -> None:
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, dims),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceNopTransposeOrPermuteWithViewPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x]
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceNopTransposeOrPermuteWithViewPass",
        )

        # Assert that permute op was removed, and a view op was placed instead
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default), 1
        )

    @expand(
        [
            # permutations replaced by transpose
            [(3, 4), (1, 0)],
            [(3, 4, 6), (0, 2, 1)],
        ]
    )
    @torch.no_grad()
    def test_replace_permute_with_transpose(
        self, shape: Tuple[int], dims: Tuple[int]
    ) -> None:
        x = torch.randn(shape)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.permute_copy.default,
            args=(x, dims),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplacePermuteWithTransposePass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module
        inputs = [x]
        validate(
            gm_before, graph_after_passes, inputs, "ReplacePermuteWithTransposePass"
        )

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
    ) -> None:
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


class TestReplaceWhereWithFullArgsWithWhereScalar(unittest.TestCase):
    def test_replace_aten_where_with_cadence(self) -> None:
        builder = GraphBuilder()
        cond_input = torch.randn(4, 8)
        cond = builder.placeholder("cond", cond_input)
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

        # Deepcopy before the pass
        gm_before = copy.deepcopy(original_gm)

        p = ReplaceWhereWithFullArgsWithWhereScalar()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [cond_input]
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceWhereWithFullArgsWithWhereScalar",
        )

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

    @expand(
        [
            [(4, 8), (4, 8), (4, 8), 0.0, 1.0],
            [(8,), (4, 8), (8,), 0.0, 1.0],
            [(4, 8), (8,), (8,), 0.0, 1.0],
        ]
    )
    def test_replace_aten_where_with_cadence_broadcast(
        self,
        cond_shape: Tuple[int],
        a_shape: Tuple[int],
        b_shape: Tuple[int],
        val1: float,
        val2: float,
    ) -> None:
        builder = GraphBuilder()
        cond_input = torch.randn(cond_shape)
        cond = builder.placeholder("cond", cond_input)
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

        # Deepcopy before the pass
        gm_before = copy.deepcopy(original_gm)

        p = ReplaceWhereWithFullArgsWithWhereScalar()
        result = cast(PassResult, p(original_gm))
        # Broadcast case should not be replaced
        self.assertFalse(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy (should be same since not modified)
        inputs = [cond_input]
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceWhereWithFullArgsWithWhereScalar",
        )

        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.where.self,
            ),
            1,
        )

    def test_replace_split_with_sizes_with_slice(self) -> None:
        builder = GraphBuilder()
        x_input = torch.randn(1, 16, 8, 4)
        x = builder.placeholder("x", x_input)
        split = builder.call_operator(
            exir_ops.edge.aten.split_with_sizes_copy.default, (x, [8, 8], 1)
        )
        # We need the outputs to be gathered by getitem ops
        out0 = builder.call_operator(operator.getitem, (split, 0))
        out1 = builder.call_operator(operator.getitem, (split, 1))
        builder.output([out0, out1])
        graph_module = builder.get_graph_module()

        gm_before = copy.deepcopy(graph_module)
        p = ReplaceSplitWithSlicePass()
        result = cast(PassResult, p(graph_module))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        validate(
            gm_before,
            graph_after_passes,
            [x_input],
            "ReplaceSplitWithSlicePass",
        )

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

    @expand([[2], [3], [4]])
    def test_replace_pow_with_mul(self, exponent: int) -> None:
        x_input = torch.randn(2, 1, 64)
        x = x_input
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten.pow.Tensor_Scalar,
            args=(x, exponent),
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplacePowWithMulPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        validate(gm_before, graph_after_passes, [x_input], "ReplacePowWithMulPass")

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

    @expand(
        [
            [1],
            [1.5],
        ]
    )
    def test_replace_pow_with_mul_not_applied(self, exponent: float) -> None:
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
    def test_no_replacement_for_conv(self) -> None:
        # Create a graph with a single im2row node.
        x = torch.randn(1, 3, 224, 224)
        pad_value = torch.tensor(0, dtype=torch.int32)
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

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        # Apply replacement pass.
        p = ReplaceIm2RowWithViewPass()
        result = p.call(gm)
        self.assertFalse(result.modified)
        gm_after_replacement = result.graph_module

        # Validate numerical accuracy
        inputs = [x, pad_value]
        validate(gm_before, gm_after_replacement, inputs, "ReplaceIm2RowWithViewPass")

        # Check that no replacement was made.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.im2row.default), 1
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.view_copy.default), 0
        )

    def test_no_replace_for_dilation(self) -> None:
        # Create a graph with a single im2row node.
        x = torch.randn(1, 3, 5, 7)
        pad_value = torch.tensor(0, dtype=torch.int32)
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

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        # Apply replacement pass.
        p = ReplaceIm2RowWithViewPass()
        result = p.call(gm)
        self.assertFalse(result.modified)
        gm_after_replacement = result.graph_module

        # Validate numerical accuracy
        inputs = [x, pad_value]
        validate(gm_before, gm_after_replacement, inputs, "ReplaceIm2RowWithViewPass")

        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.im2row.default), 1
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.view_copy.default), 0
        )

    def test_replace_linear_like_conv(self) -> None:
        # Create a graph with a single im2row node.
        in_h, in_w = 13, 15
        x = torch.randn(1, 3, in_h, in_w)
        pad_value = torch.tensor(0, dtype=torch.int32)
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

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        # Apply replacement pass.
        p = ReplaceIm2RowWithViewPass()
        result = p.call(gm)
        self.assertTrue(result.modified)
        gm_after_replacement = result.graph_module

        # Validate numerical accuracy
        inputs = [x, pad_value]
        validate(gm_before, gm_after_replacement, inputs, "ReplaceIm2RowWithViewPass")

        # In this test, the kernel width/height is the same as the input width/height.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.cadence.im2row.default), 0
        )
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.view_copy.default), 1
        )


class TestReplaceConvWithChannelLastConvPass(unittest.TestCase):
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
            op=exir_ops.edge.cadence.conv1d.default,
            args=args,
        )

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
            op=exir_ops.edge.cadence.conv2d.default,
            args=args,
        )

    def create_quantized_convolution_graph_module(
        self, channels_last: Optional[bool] = None
    ) -> tuple[tuple[torch.Tensor, ...], torch.fx.GraphModule]:
        """Helper to create a quantized conv node.

        quantized_conv_per_tensor(
            Tensor input, Tensor weight, Tensor bias, int[] stride, SymInt[] padding,
            int[] dilation, int groups, int input_zero_point, int weight_zero_point,
            Tensor bias_scale, float out_scale, int out_zero_point, int out_multiplier,
            int out_shift, bool channel_last=False) -> (Tensor Z)"
        """
        if channels_last:
            x = torch.randint(0, 100, (1, 224, 56, 3), dtype=torch.int32)
            w = torch.randint(0, 100, (16, 16, 16, 3), dtype=torch.int32)
        else:
            x = torch.randint(0, 100, (1, 3, 224, 56), dtype=torch.int32)
            w = torch.randint(0, 100, (16, 3, 16, 16), dtype=torch.int32)
        b = torch.randn(16)
        stride = (2, 2)
        padding = (0, 0)
        dilation = (1, 1)
        groups = 1
        input_zero_point = 0
        w_zero_point = 100
        b_scale = 10
        out_scale = 1
        out_zero_point = 0
        out_multiplier = 5
        out_shift = 5
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
            op = exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor
        else:
            op = exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor

        placeholders = (x, w, b)

        return placeholders, single_op_builder(
            placeholders=placeholders,
            op=op,
            args=args,
        )

    def test_quantized_convolution_default_channel_last(self) -> None:
        # Create a graph with a single convolution node.
        placeholders, gm = self.create_quantized_convolution_graph_module()
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv2d_nchw.per_tensor), 1
        )
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)

        # Apply replacement pass.
        p = ReplaceConvWithChannelLastConvPass()
        original = copy.deepcopy(gm)
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(
                gm_after_replacement,
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
            ),
            1,
        )
        # Three permutes are added, two for the input/weights and one for the output.
        self.assertEqual(
            count_node(gm_after_replacement, exir_ops.edge.aten.permute_copy.default),
            3,
        )
        validate(
            gm_after_replacement,
            original,
            placeholders,
            "ReplaceConvWithChannelLastConvPass",
        )

    def test_no_transpose_if_already_quantized_conv_channel_last(self) -> None:
        # Create a graph with a single im2row node.
        placeholders, gm = self.create_quantized_convolution_graph_module(
            channels_last=True
        )
        # Check if graph module is valid by running exportpass on it.
        original = copy.deepcopy(gm)
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor), 1
        )

        # Apply replacement pass.
        p = ReplaceConvWithChannelLastConvPass()
        gm_after_replacement = p.call(gm).graph_module
        # Check that no replacement was made.
        self.assertEqual(
            count_node(
                gm_after_replacement,
                exir_ops.edge.cadence.quantized_conv2d_nhwc.per_tensor,
            ),
            1,
        )
        self.assertEqual(count_node(gm, exir_ops.edge.aten.permute_copy.default), 0)
        validate(
            gm_after_replacement,
            original,
            placeholders,
            "ReplaceConvWithChannelLastConvPass",
        )


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

    def test_slice_no_transpose_if_already_outermost(self) -> None:
        # Create a graph with a single slice node.
        gm = self.create_slice_graph((3, 224, 224), 0, 1, 2)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_copy.Tensor), 1)

        # Apply replacement pass.
        p = MakeSliceAndCatDimOutermostPass()
        gm_after_pass = cast(PassResult, p(gm)).graph_module

        # Assert that no transpose ops were added.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_slice_no_transpose_if_outermost_dimensions_are_one(self) -> None:
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_slice_graph((1, 3, 4, 6), 1, 1, 2)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_copy.Tensor), 1)

        # Apply replacement pass.
        p = MakeSliceAndCatDimOutermostPass()
        gm_after_pass = cast(PassResult, p(gm)).graph_module

        # Assert that no transpose ops were added. The slice is on the second
        # outermost dimension, but the outermost dimension is already 1.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_slice_insert_transpose(self) -> None:
        # Create a graph with a single slice node.
        gm = self.create_slice_graph((1, 3, 4, 6), 2, 1, 2)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.slice_copy.Tensor), 1)

        # Apply replacement pass.
        p = MakeSliceAndCatDimOutermostPass()
        gm_after_pass = cast(PassResult, p(gm)).graph_module

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

    def test_cat_no_transpose_if_already_outermost(self) -> None:
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_cat_graph(input_shapes=((1, 3, 5), (2, 3, 5)), cat_dim=0)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.cat.default), 1)

        # Apply replacement pass.
        p = MakeSliceAndCatDimOutermostPass()
        gm_after_pass = cast(PassResult, p(gm)).graph_module

        # Assert that no transpose ops were added. The slice is on the second
        # outermost dimension, but the outermost dimension is already 1.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_cat_no_transpose_if_outermost_dimensions_are_one(self) -> None:
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_cat_graph(input_shapes=((1, 1, 3, 5), (1, 2, 3, 5)), cat_dim=1)
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.cat.default), 1)

        # Apply replacement pass.
        p = MakeSliceAndCatDimOutermostPass()
        gm_after_pass = cast(PassResult, p(gm)).graph_module

        # Assert that no transpose ops were added. The slice is on the second
        # outermost dimension, but the outermost dimension is already 1.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            0,
        )

    def test_cat_insert_transpose(self) -> None:
        # Create a graph with a single slice node on second outermost dimension.
        gm = self.create_cat_graph(
            input_shapes=((1, 1, 3, 5), (1, 1, 3, 3)), cat_dim=-1
        )
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module
        self.assertEqual(count_node(gm, exir_ops.edge.aten.cat.default), 1)

        # Apply replacement pass.
        p = MakeSliceAndCatDimOutermostPass()
        gm_after_pass = cast(PassResult, p(gm)).graph_module

        # Assert that transpose ops were added to make cat on outermost dimension.
        self.assertEqual(
            count_node(gm_after_pass, exir_ops.edge.aten.transpose_copy.int),
            3,
        )


class TestReplaceEmptyTensorsWithFullPass(unittest.TestCase):
    def _get_slice_empty_gm(self) -> tuple[torch.fx.GraphModule, torch.Tensor]:
        builder = GraphBuilder()
        x_input = torch.randn(4)
        x = builder.placeholder("x", x_input)
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
        return builder.get_graph_module(), x_input

    def test_empty_slice(self) -> None:
        gm, x_input = self._get_slice_empty_gm()
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

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        result = ReplaceEmptyTensorsWithFullPass().call(gm)
        self.assertTrue(result.modified)
        updated_gm = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate(gm_before, updated_gm, inputs, "ReplaceEmptyTensorsWithFullPass")

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

    @expand(
        [
            ("int", int(123)),
            ("float", float(456.0)),
        ],
    )
    @torch.no_grad()
    def test_extract_mul_argument_to_full(
        self, _: str, value: Union[int, float]
    ) -> None:
        if isinstance(value, int):
            x_input = torch.randint(0, 100, (1,), dtype=torch.int32)
        else:
            x_input = torch.randn((1,), dtype=torch.float32)

        gm = single_op_builder(
            placeholders=(x_input,),
            op=torch.ops.aten.mul.Tensor,
            args=(x_input, value),
            kwargs={},
        )

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        p = ReplaceMulTensorWithMulAndFullOpsPass()
        result = p.call(gm)
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceMulTensorWithMulAndFullOpsPass",
        )

        self.assertTrue(
            op_counts_match(
                graph_after_passes,
                expected_op_counts={
                    torch.ops.aten.mul.Tensor: 1,
                    torch.ops.aten.full.default: 1,
                },
            )
        )


class TestReplaceAdaptiveAvgPoolWithAtenAvgPoolPass(unittest.TestCase):
    def _get_adaptive_avg_pool_gm(
        self, input_shape: Tuple[int, int, int, int], output_shape: Tuple[int, int]
    ) -> tuple[torch.Tensor, torch.fx.GraphModule]:
        builder = GraphBuilder()
        x_input = torch.randn(*input_shape)
        x = builder.placeholder("x", x_input)
        adaptive_avg_pool2d = builder.call_operator(
            exir_ops.edge.aten._adaptive_avg_pool2d.default, (x, output_shape)
        )
        builder.output([adaptive_avg_pool2d])
        return x_input, builder.get_graph_module()

    def test_replace_adaptive_avg_pool_with_aten_avg_pool(self) -> None:
        x_input, gm = self._get_adaptive_avg_pool_gm((1, 64, 128, 128), (8, 8))
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.aten.avg_pool2d.default,
                )
            ),
            0,
        )

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        p = ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass()
        result = p.call(gm)
        self.assertTrue(result.modified)
        updated_gm = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate(
            gm_before,
            updated_gm,
            inputs,
            "ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass",
        )

        self.assertEqual(
            len(
                updated_gm.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
                )
            ),
            0,
        )
        avg_pool2d_nodes = updated_gm.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.avg_pool2d.default
        )
        self.assertEqual(
            len(avg_pool2d_nodes),
            1,
        )
        avg_pool2d_node = avg_pool2d_nodes[0]

        self.assertEqual(avg_pool2d_node.args[1], [16, 16])  # kernel_size is 16x16
        self.assertEqual(avg_pool2d_node.args[2], [16, 16])  # stride is 16, 16
        self.assertEqual(avg_pool2d_node.args[3], [0, 0])  # padding is 0, 0
        self.assertEqual(avg_pool2d_node.args[4], False)  # ceil_mode is False
        self.assertEqual(avg_pool2d_node.args[5], True)  # count_include_pad is True
        self.assertEqual(avg_pool2d_node.args[6], None)  # divisor_override is None

    def test_replace_adaptive_avg_pool_with_aten_avg_pool_irregular(self) -> None:
        x_input, gm = self._get_adaptive_avg_pool_gm((1, 64, 128, 128), (9, 9))
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                gm.graph.find_nodes(
                    op="call_function", target=exir_ops.edge.aten.avg_pool2d.default
                )
            ),
            0,
        )

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        # Shapes are not multiples of each other, so pass will not trigger
        p = ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass()
        result = p.call(gm)
        self.assertFalse(result.modified)
        updated_gm = result.graph_module

        # Validate numerical accuracy (should be same since not modified)
        inputs = [x_input]
        validate(
            gm_before,
            updated_gm,
            inputs,
            "ReplaceAdaptiveAvgPoolWithAtenAvgPoolPass",
        )

        self.assertEqual(
            len(
                updated_gm.graph.find_nodes(
                    op="call_function",
                    target=exir_ops.edge.aten._adaptive_avg_pool2d.default,
                )
            ),
            1,
        )
        avg_pool2d_nodes = updated_gm.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.avg_pool2d.default
        )
        self.assertEqual(
            len(avg_pool2d_nodes),
            0,
        )


class TestReplaceAtenAvgPoolWithCadenceAvgPoolPass(unittest.TestCase):
    def _get_aten_avg_pool1d_gm(
        self, input_shape: Tuple[int, int, int], kernel_size: int
    ) -> tuple[torch.Tensor, torch.fx.GraphModule]:
        builder = GraphBuilder()
        x_input = torch.randn(*input_shape)
        x = builder.placeholder("x", x_input)
        avg_pool1d = builder.call_operator(
            exir_ops.edge.aten.avg_pool1d.default, (x, [kernel_size])
        )
        builder.output([avg_pool1d])
        return x_input, builder.get_graph_module()

    def _get_aten_avg_pool2d_gm(
        self, input_shape: Tuple[int, int, int, int], kernel_size: Tuple[int, int]
    ) -> tuple[torch.Tensor, torch.fx.GraphModule]:
        builder = GraphBuilder()
        x_input = torch.randn(*input_shape)
        x = builder.placeholder("x", x_input)
        avg_pool2d = builder.call_operator(
            exir_ops.edge.aten.avg_pool2d.default, (x, list(kernel_size))
        )
        builder.output([avg_pool2d])
        return x_input, builder.get_graph_module()

    def test_replace_aten_avg_pool1d_with_cadence(self) -> None:
        x_input, gm = self._get_aten_avg_pool1d_gm((1, 32, 64), 3)
        self.assertEqual(
            count_node(gm, exir_ops.edge.aten.avg_pool1d.default),
            1,
        )
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.avg_pool2d.default),
            0,
        )

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        p = ReplaceAtenAvgPoolWithCadenceAvgPoolPass()
        result = p.call(gm)
        self.assertTrue(result.modified)
        updated_gm = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate(
            gm_before,
            updated_gm,
            inputs,
            "ReplaceAtenAvgPoolWithCadenceAvgPoolPass",
        )

        # avg_pool1d should be replaced with view operations and avg_pool2d
        self.assertEqual(
            count_node(updated_gm, exir_ops.edge.aten.avg_pool1d.default),
            0,
        )
        self.assertEqual(
            count_node(updated_gm, exir_ops.edge.cadence.avg_pool2d.default),
            1,
        )
        # Should have view operations for reshaping
        self.assertGreater(
            count_node(updated_gm, exir_ops.edge.aten.view_copy.default),
            0,
        )

    def test_replace_aten_avg_pool2d_with_cadence(self) -> None:
        x_input, gm = self._get_aten_avg_pool2d_gm((1, 32, 64, 64), (3, 3))
        self.assertEqual(
            count_node(gm, exir_ops.edge.aten.avg_pool2d.default),
            1,
        )
        self.assertEqual(
            count_node(gm, exir_ops.edge.cadence.avg_pool2d.default),
            0,
        )

        # Deepcopy before the pass
        gm_before = copy.deepcopy(gm)

        p = ReplaceAtenAvgPoolWithCadenceAvgPoolPass()
        result = p.call(gm)
        self.assertTrue(result.modified)
        updated_gm = result.graph_module

        # Validate numerical accuracy
        inputs = [x_input]
        validate(
            gm_before,
            updated_gm,
            inputs,
            "ReplaceAtenAvgPoolWithCadenceAvgPoolPass",
        )

        # avg_pool2d should be replaced with cadence avg_pool2d
        self.assertEqual(
            count_node(updated_gm, exir_ops.edge.aten.avg_pool2d.default),
            0,
        )
        self.assertEqual(
            count_node(updated_gm, exir_ops.edge.cadence.avg_pool2d.default),
            1,
        )


class TestReplaceLinalgSvdPass(unittest.TestCase):
    @expand(
        [
            ("2x2", (2, 2)),
            ("3x3", (3, 3)),
            ("4x5", (4, 5)),
            ("10x10", (10, 10)),
        ]
    )
    @torch.no_grad()
    def test_replace_aten_linalg_svd_with_cadence_linalg_svd(
        self, _: str, shape: Tuple[int, int]
    ) -> None:
        x = torch.randn(shape, dtype=torch.float32)
        original_gm = single_op_builder(
            placeholders=(x,),
            op=exir_ops.edge.aten._linalg_svd.default,
            args=(x, False, True),
            kwargs={"driver": None},
        )

        p = ReplaceAtenLinalgSvdWithCadenceLinalgSvdPass()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Assert that the aten linalg_svd op was replaced with cadence linalg_svd op
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten._linalg_svd.default),
            0,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.cadence.linalg_svd.default),
            1,
        )

    @expand([("dtype",), ("default",)])
    @torch.no_grad()
    def test_replace_quantized_embedding(
        self,
        name: str,
    ) -> None:
        embedding = torch.ones(5, 6, dtype=torch.int8)
        indices = torch.tensor([0, 2], dtype=torch.int32)
        scales = torch.ones(5, 2, dtype=torch.float32)
        zero_points = None

        original_gm = single_op_builder(
            placeholders=(embedding, scales, indices),
            op=(
                exir_ops.edge.quantized_decomposed.embedding_byte.dtype
                if name == "dtype"
                else exir_ops.edge.quantized_decomposed.embedding_byte.default
            ),
            args=(embedding, scales, zero_points, -128, 127, indices),
            kwargs={"dtype": torch.float32} if name == "dtype" else {},
        )

        gm_before = copy.deepcopy(original_gm)
        p = ReplaceTorchQuantizedEmbeddingWithCadenceQuantizedEmbedding()
        result = cast(PassResult, p(original_gm))
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Validate numerical accuracy
        inputs = [embedding, scales, indices]
        validate(
            gm_before,
            graph_after_passes,
            inputs,
            "ReplaceTorchQuantizedEmbeddingWithCadenceQuantizedEmbedding",
        )

        self.assertEqual(
            count_node(
                graph_after_passes,
                (
                    exir_ops.edge.quantized_decomposed.embedding_byte.dtype
                    if name == "dtype"
                    else exir_ops.edge.quantized_decomposed.embedding_byte.default
                ),
            ),
            0,
        )

        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.cadence.quantized_embedding_byte.default,
            ),
            1,
        )


class TestReplaceLogicalNotBooleanWhereWithWherePass(unittest.TestCase):
    """Tests for the ReplaceLogicalNotBooleanWhereWithWherePass."""

    def test_replace_where_with_logical_not_boolean(self) -> None:
        """Test that where(logical_not(bool_cond), x, y) is replaced with where(bool_cond, y, x)."""
        # Setup: Create a graph with where(logical_not(bool_cond), x, y)
        builder = GraphBuilder()
        bool_cond_ = torch.randn(4, 8) > 0
        x_ = torch.randn(4, 8)
        y_ = torch.randn(4, 8)

        bool_cond = builder.placeholder("bool_cond", bool_cond_)
        x = builder.placeholder("x", x_)
        y = builder.placeholder("y", y_)

        # Create logical_not node
        logical_not = builder.call_operator(
            op=exir_ops.edge.aten.logical_not.default,
            args=(bool_cond,),
        )

        # Create where node using logical_not
        where_node = builder.call_operator(
            op=exir_ops.edge.aten.where.self,
            args=(logical_not, x, y),
        )
        builder.output([where_node])
        original_gm = builder.get_graph_module()

        # Make a copy of the original graph before applying the pass
        original_gm_copy = copy.deepcopy(original_gm)

        # Execute: Apply the replacement pass
        p = ReplaceLogicalNotBooleanWhereWithWherePass()
        result = cast(PassResult, p(original_gm))

        # Assert: Verify the pass modified the graph
        self.assertTrue(result.modified)
        graph_after_passes = result.graph_module

        # Assert: Verify logical_not is removed (dead code elimination)
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.logical_not.default),
            0,
        )

        # Assert: Verify where node still exists
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.where.self),
            1,
        )

        # Assert: Verify the arguments are flipped (condition uses original bool_cond, x and y are swapped)
        where_nodes = list(
            graph_after_passes.graph.find_nodes(
                op="call_function", target=exir_ops.edge.aten.where.self
            )
        )
        for node in where_nodes:
            # First arg should be the original bool_cond (not the logical_not)
            self.assertEqual(node.args[0].name, "bool_cond")
            # Second and third args should be swapped (y, x instead of x, y)
            self.assertEqual(node.args[1].name, "y")
            self.assertEqual(node.args[2].name, "x")

        # Assert: Verify outputs match exactly by running both graphs
        validate(
            original_gm_copy,
            graph_after_passes,
            (bool_cond_, x_, y_),
            "ReplaceLogicalNotBooleanWhereWithWherePass",
        )

    def test_no_replacement_without_logical_not(self) -> None:
        """Test that the pass does NOT apply when there's no logical_not."""
        # Setup: Create a graph with where(bool_cond, x, y) without logical_not
        builder = GraphBuilder()
        bool_cond = builder.placeholder("bool_cond", torch.randn(4, 8) > 0)
        x = builder.placeholder("x", torch.randn(4, 8))
        y = builder.placeholder("y", torch.randn(4, 8))

        # Create where node directly without logical_not
        where_node = builder.call_operator(
            op=exir_ops.edge.aten.where.self,
            args=(bool_cond, x, y),
        )
        builder.output([where_node])
        original_gm = builder.get_graph_module()

        # Execute: Apply the replacement pass
        p = ReplaceLogicalNotBooleanWhereWithWherePass()
        result = cast(PassResult, p(original_gm))

        # Assert: Verify the pass did NOT modify the graph
        self.assertFalse(result.modified)
        graph_after_passes = result.graph_module

        # Assert: Verify where node still exists unchanged
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.where.self),
            1,
        )

        for node in graph_after_passes.graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.where.self
        ):
            self.assertEqual(node.args[0].name, "bool_cond")
            self.assertEqual(node.args[1].name, "x")
            self.assertEqual(node.args[2].name, "y")
