# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.compiler import export_to_edge, quantize_pt2
from executorch.backends.cadence.aot.graph_builder import single_op_builder
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.backends.cadence.aot.replace_ops import (
    ForceChannelLastForConvPass,
    MakeSliceAndCatDimOutermostPass,
    ReplaceAddMMWithLinearPass,
    ReplaceAtenConvolutionWithJarvisConvolutionPass,
    ReplaceAtenLinalgVectorNormWithCadenceLinalgVectorNormPass,
    ReplaceConstantPadNdWithSlicePass,
    ReplaceConvolutionOptionalArgsWithConcreteArgsPass,
    ReplaceConvWithIm2RowAndLinear,
    ReplaceFunctionallyEquivalentOpTargets,
    ReplaceIm2RowWithViewPass,
    ReplaceLinearWithFullyConnectedOpPass,
    ReplaceMMWithAddMMPass,
    ReplaceNopTransposeOrPermuteWithViewPass,
    ReplacePadWithCatPass,
    ReplacePermuteWithTransposePass,
    ReplaceRepeatWithCatPass,
    ReplaceScalarTensorWithFullPass,
    ReplaceScalarWithTensorArgPass,
    ReplaceSelectWithViewOpPass,
    ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass,
    ReplaceSqueezeAndUnsqueezeWithViewPass,
    ReplaceTCopyWithTransposePass,
    ReplaceTransposedConvWithLinearPass,
    ReplaceTrivialConvWithLinear,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes import dead_code_elimination_pass

from parameterized.parameterized import parameterized
from torch._ops import OpOverload
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
            [(3, 5), (0, 0)],
            [
                (20, 1, 80),
                (0, 0),
            ],
        ]
    )
    @torch.no_grad()
    def test_replace_constant_pad_nd_with_slice(
        self, shape: Tuple[int], padding: Tuple[int]
    ):
        # F.pad is converted to aten::constant_pad_nd after functionalization & decomposition.
        class Padding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.padding = padding

            def forward(self, x: torch.Tensor):
                return F.pad(x, self.padding)

        model = Padding()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplaceConstantPadNdWithSlicePass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
            [(7, 5, 6), 1.23],
            [(7, 5), 2],
        ]
    )
    @torch.no_grad()
    def test_add_replace_scalar_with_tensor_arg(self, shape: Tuple[int], other: float):
        class Add(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.add.Scalar(x, other)

        model = Add()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplaceScalarWithTensorArgPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
            [(7, 5, 6), 1.23],
            [(7, 5), 2],
            [(10), 42949],
        ]
    )
    @torch.no_grad()
    def test_sub_replace_scalar_with_tensor_arg(self, shape: Tuple[int], other: float):
        class Sub(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.sub.Scalar(x, other)

        model = Sub()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplaceScalarWithTensorArgPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
            [(7, 5, 6), 1.23],
            [(7, 5), 2],
            [(513), 3],
        ]
    )
    @torch.no_grad()
    def test_mul_replace_scalar_with_tensor_arg(self, shape: Tuple[int], other: float):
        class Mul(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.mul.Scalar(x, other)

        model = Mul()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplaceScalarWithTensorArgPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
            [(7, 5, 6), 1.23],
            [(7, 5), 2],
        ]
    )
    @torch.no_grad()
    def test_div_replace_scalar_with_tensor_arg(
        self,
        shape: Tuple[int],
        other: float,
    ):
        class Div(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.div.Scalar(x, other)

        model = Div()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplaceScalarWithTensorArgPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
            [(2, 3, 5, 6)],
            [(7, 6, 5)],
            [(4, 4)],
            [(316)],
        ]
    )
    @torch.no_grad()
    def test_replace_functionally_equivalent_op_targets_relu(self, shape: Tuple[int]):
        model = torch.nn.ReLU()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        p = ReplaceFunctionallyEquivalentOpTargets()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.relu.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.relu_.default),
            0,
        )

    @parameterized.expand(
        [
            # split the only dimension
            [(50,), i, 0]
            for i in range(2, 7)
        ]
        + [
            # split the leading dim
            [(10, 2, 3), i, 0]
            for i in range(2, 7)
        ]
        + [
            # split the trailing dim
            [(3, 3, 6), i, 2]
            for i in range(2, 6)
        ]
        + [
            # split the dim in the middle
            [(3, 5, 14, 2, 3), i, 2]
            for i in range(2, 7)
        ]
    )
    @torch.no_grad()
    def test_replace_functionally_equivalent_op_targets_unsafe_split(
        self, shape: Tuple[int], split_size: int, dim: int
    ):
        class TensorSplitWithSizes(torch.nn.Module):
            def __init__(self, split_size: int, dim: int, op: OpOverload):
                super().__init__()
                self.split_size = split_size
                self.dim = dim
                self.op = op

            def forward(self, x: torch.Tensor):
                return self.op(x, self.split_size, self.dim)

        x = torch.randn(shape)
        model = TensorSplitWithSizes(split_size, dim, torch.unsafe_split)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        p = ReplaceFunctionallyEquivalentOpTargets()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.aten.split_with_sizes_copy.default
            ),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.unsafe_split.Tensor),
            0,
        )

    @parameterized.expand(
        [
            [(16, 32)],
            [(1, 240)],
            [(4, 16)],
        ]
    )
    @torch.no_grad()
    def test_replace_t_copy_with_transpose(self, shape: Tuple[int]):
        class TCopy(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return exir_ops.edge.aten.t_copy(x)

        w = torch.randn(shape)
        inputs = (w,)
        p1 = ReplaceTCopyWithTransposePass()
        p2 = ReplacePermuteWithTransposePass()
        model = TCopy()
        graph_module = export_to_edge(model, inputs).exported_program().graph_module
        graph_after_passes = cast(
            PassResult, p2(cast(PassResult, p1(graph_module)).graph_module)
        ).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.t_copy),
            0,
        )

    @parameterized.expand(
        [
            [(1, 8, 33), 8, 16, 3],
            [(1, 8, 33), 8, 16, 5, 2],
            [(1, 8, 33), 8, 16, 3, 2, 4, 3, False, False, False],
            # channel last
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
        bias: bool = True,
        channel_last: bool = False,
    ):
        class TConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tconv1d = torch.nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=in_channels if depthwise else 1,
                    bias=bias,
                )

            def forward(self, x: torch.Tensor):
                if channel_last:
                    x = x.permute([0, 2, 1])
                x = self.tconv1d(x)
                if channel_last:
                    x = x.permute([0, 2, 1])
                return x

        x = torch.randn(shape)
        model = TConv()
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        p1 = ReplaceAtenConvolutionWithJarvisConvolutionPass()
        p2 = ReplaceTransposedConvWithLinearPass()
        graph_after_passes = cast(
            PassResult, p2(cast(PassResult, p1(graph_module)).graph_module)
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
        bias: bool = True,
        channel_last: bool = False,
    ):
        class Conv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=in_channels if depthwise else 1,
                    bias=bias,
                )

            def forward(self, x: torch.Tensor):
                if channel_last:
                    x = x.permute([0, 2, 1])
                x = self.conv1d(x)
                if channel_last:
                    x = x.permute([0, 2, 1])
                return x

        x = torch.randn(shape)
        model = Conv()

        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplaceConvolutionOptionalArgsWithConcreteArgsPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
            [(1, 2, 3), (1, 1)],
            [
                (20, 1, 80),
                (1, 4),
            ],
        ]
    )
    @torch.no_grad()
    def test_replace_pad_with_cat(self, shape: Tuple[int], padding: Tuple[int]):
        # F.pad is converted to aten::constant_pad_nd after functionalization & decomposition.
        class Padding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.padding = padding

            def forward(self, x: torch.Tensor):
                return F.pad(x, self.padding)

        model = Padding()
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplacePadWithCatPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.cat.default),
            1,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.pad.default),
            0,
        )

    @torch.no_grad()
    def test_replace_repeat_with_cat(self):
        class Repeat(torch.nn.Module):
            def forward(self, x):
                x1 = torch.add(x, 2.4, 3.1)
                return torch.ops.aten.repeat(x1, [1, 2])

        x = torch.ones(3, 5)
        graph_module = export_to_edge(Repeat(), (x,)).exported_program().graph_module

        p = ReplaceRepeatWithCatPass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
            [(36, 1, 2, 80), (1)],
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
        class MaskedFill(torch.nn.Module):
            def __init__(self, value: float):
                super().__init__()
                self.value = value

            def forward(self, x: torch.Tensor, mask: torch.Tensor):
                return torch.masked_fill(x, mask, self.value)

        x = torch.randn(shape)
        mask = torch.randn(mask_shape if mask_shape else shape) > 0
        value = 0.5 * torch.mean(x).item()
        model = MaskedFill(value)
        graph_module = export_to_edge(model, (x, mask)).exported_program().graph_module

        p = ReplaceScalarTensorWithFullPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.full.default),
            1,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.where.self),
            1,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.masked_fill),
            0,
        )

    @parameterized.expand(
        [
            [(1), 1.5],
            [(1), 0.0],
        ]
    )
    @torch.no_grad()
    def test_replace_scalar_tensor_with_full(self, shape: Tuple[int], value: float):
        class ScalarTensor(torch.nn.Module):
            def __init__(self, shape: Tuple[int], value: float):
                super().__init__()
                self.shape = shape
                self.value = value

            def forward(self, x: torch.Tensor):
                return torch.scalar_tensor(value)

        model = ScalarTensor(shape, value)
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        p = ReplaceScalarTensorWithFullPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
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
        shape, in_features, out_features, bias = (1, 14), 14, 128, False
        model = torch.nn.Linear(in_features, out_features, bias=bias)
        x = torch.randn(shape)

        graph_module = export_to_edge(model, (x,)).exported_program().graph_module
        permute_to_trans_pass = ReplacePermuteWithTransposePass()
        mm_to_addmm_pass = ReplaceMMWithAddMMPass()
        add_to_linear_pass = ReplaceAddMMWithLinearPass()
        linear_to_fullyconnected_pass = ReplaceLinearWithFullyConnectedOpPass()
        graph_after_passes = linear_to_fullyconnected_pass(
            add_to_linear_pass(
                mm_to_addmm_pass(
                    permute_to_trans_pass(graph_module).graph_module
                ).graph_module
            ).graph_module
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
        class AddMM(torch.nn.Module):
            def __init__(self, alpha: float = 1, beta: float = 1):
                super().__init__()
                self.alpha = alpha
                self.beta = beta

            def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
                return torch.addmm(
                    x, y, z.transpose(1, 0), alpha=self.alpha, beta=self.beta
                )

        # alpha, beta must be 1 to be 1 to enable ReplaceAddMMWithLinearPass
        # get_attr will always turn into placeholders and mutable outputs in PT2
        M, K, N, alpha, beta = 14, 48, 24, 1.0, 1.0
        x = torch.randn(N)
        y = torch.randn(M, K)
        z = torch.randn(N, K)

        # test addmm
        model = AddMM(alpha=alpha, beta=beta)
        graph_module = export_to_edge(model, (x, y, z)).exported_program().graph_module

        tp = ReplacePermuteWithTransposePass()
        ap = ReplaceAddMMWithLinearPass()
        graph_after_passes = cast(
            PassResult, ap(cast(PassResult, tp(graph_module)).graph_module)
        ).graph_module
        self.assertIsNotNone(graph_after_passes)

        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.addmm.default),
            1,
        )

        # Assert that all the aten.addmm nodes are removed.
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
        # The mm ops will be convereted to addmm ops by Jarvis
        class MM(torch.nn.Module):
            def __init__(self, K, N):
                super().__init__()
                self.K = K
                self.N = N

            def forward(self, y: torch.Tensor, z: torch.Tensor):
                return torch.ops.aten.mm(y, z)

        M, K, N = 14, 48, 24
        y = torch.randn(M, K)
        z = torch.randn(K, N)

        # test addmm
        model = MM(K, N)
        graph_module = export_to_edge(model, (y, z)).exported_program().graph_module

        # First, replace the aten.mm with an aten.addmm op
        p = ReplaceMMWithAddMMPass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertIsNotNone(graph_after_passes)

        # Assert that all the aten.mm nodes are removed.
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.addmm.default),
            1,
        )

        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mm),
            0,
        )

    @parameterized.expand(
        [
            # shape
            [(5, 1, 6, 7)],
            [(1)],
            [(4, 3, 2)],
            # shape, dim to squeeze
            [(2, 1), 0],
            [(2, 7, 1, 3), 1],
            [(2, 1, 3), 2],
        ]
    )
    @torch.no_grad()
    def test_replace_squeeze_with_view(self, shape: Tuple[int], dim=None):
        # The squeeze ops will be convereted to view ops by Jarvis
        class Squeeze(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x: torch.Tensor):
                if self.dim is None:
                    return torch.squeeze(x)
                return torch.squeeze(x, self.dim)

        model = Squeeze(dim)
        x = torch.randn(shape)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        # First, replace the aten.squeeze_copy with an aten.view_copy op
        p = ReplaceSqueezeAndUnsqueezeWithViewPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertIsNotNone(graph_after_passes)

        # Assert that all the aten.squeeze_copy nodes are removed.
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.aten.squeeze_copy),
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
        class Unsqueeze(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x: torch.Tensor):
                return torch.unsqueeze(x, self.dim)

        # Test that the pass works for all dims.
        model = Unsqueeze(dim)
        x = torch.randn(5, 6, 7)
        graph_module = export_to_edge(model, (x,)).exported_program().graph_module

        # First, replace the aten.unsqueeze_copy with an aten.view_copy op
        p = ReplaceSqueezeAndUnsqueezeWithViewPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module
        self.assertIsNotNone(graph_after_passes)

        # Assert that all the aten.unsqueeze_copy nodes are removed.
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.view_copy.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.aten.unsqueeze_copy),
            0,
        )

    @torch.no_grad()
    def test_replace_single_element_tensor_arguments_from_full_op_with_scalar(
        self,
        in_features: int = 16,
        out_features: int = 16,
    ):
        # Tensors - these will be inputs to graph.
        x = torch.randn([1, in_features])

        inputs = (x,)
        model = torch.nn.Linear(in_features=in_features, out_features=out_features)
        quantized_model = quantize_pt2(model, inputs)

        exported_program = export_to_edge(quantized_model, inputs).exported_program()

        # By default, the quantized linear op should have constant scalar attributes.
        self.assertTargetCountsEqual(
            exported_program.graph_module,
            [
                # One quantized linear op.
                (exir_ops.edge.cadence.quantized_linear.default, 1),
                # No per tensor quantized linear ops.
                (exir_ops.edge.cadence.quantized_linear.per_tensor, 0),
                # Three aten.full ops.
                (exir_ops.edge.aten.full.default, 3),
            ],
        )

        # Apply replacement pass.
        p = ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass()
        graph_after_passes = p(exported_program.graph_module)
        self.assertIsNotNone(graph_after_passes)
        gm = dead_code_elimination_pass(graph_after_passes.graph_module).graph_module

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
        # Tensors - these will be inputs to graph.
        x = torch.randn([1, in_features])

        inputs = (x,)
        model = torch.nn.Linear(in_features=in_features, out_features=out_features)
        quantized_model = quantize_pt2(model, inputs)

        exported_program = export_to_edge(quantized_model, inputs).exported_program()

        # By default, the quantized linear op should have constant scalar attributes.
        self.assertTargetCountsEqual(
            exported_program.graph_module,
            [
                # One quantized linear op.
                (exir_ops.edge.cadence.quantized_linear.default, 1),
                # No per tensor quantized linear ops.
                (exir_ops.edge.cadence.quantized_linear.per_tensor, 0),
                # Three aten.full ops.
                (exir_ops.edge.aten.full.default, 3),
            ],
        )

        for node in exported_program.graph_module.graph.nodes:
            # Replace the `shape` argument for aten.full op with a tuple.
            if node.target == exir_ops.edge.aten.full.default:
                node.args = (tuple(node.args[0]), node.args[1])

        # Apply replacement pass.
        p = ReplaceSingleElementTensorArgumentsFromFullOpWithScalarPass()
        graph_after_passes = p(exported_program.graph_module)
        self.assertIsNotNone(graph_after_passes)
        gm = dead_code_elimination_pass(graph_after_passes.graph_module).graph_module

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
        class Conv(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, kernel_size: int):
                super().__init__()
                self.conv1d = torch.nn.Conv1d(in_features, out_features, kernel_size)

            def forward(self, x):
                return self.conv1d(x)

        model_conv1d = Conv(96, 192, 7)
        x = torch.randn(1, 96, 7)
        graph_module = (
            export_to_edge(model_conv1d, (x,)).exported_program().graph_module
        )

        # First, replace the aten convolution with a cadence.convolution op
        p1 = ReplaceAtenConvolutionWithJarvisConvolutionPass()
        temp_graph = p1(graph_module).graph_module
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
        class Conv(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, kernel_size: int):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(in_features, out_features, kernel_size)

            def forward(self, x):
                return self.conv2d(x)

        model_conv2d = Conv(96, 192, 7)
        x = torch.randn(1, 96, 7, 7)
        graph_module = (
            export_to_edge(model_conv2d, (x,)).exported_program().graph_module
        )

        # First, replace the aten convolution with a cadence.convolution op
        p1 = ReplaceAtenConvolutionWithJarvisConvolutionPass()
        temp_graph = p1(graph_module).graph_module
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
        class Conv(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int, kernel_size: int):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(in_features, out_features, kernel_size)

            def forward(self, x):
                return self.conv2d(x)

        model_conv2d = Conv(96, 192, 7)
        x = torch.randn(1, 96, 47, 37)
        graph_module = (
            compiler.export_to_cadence(model_conv2d, (x,))
            .exported_program()
            .graph_module
        )

        p = ReplaceConvWithIm2RowAndLinear()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

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
        class Select(torch.nn.Module):
            def forward(self, x):
                return x.select(dim, index)

        x = torch.randn(shape)
        graph_module = export_to_edge(Select(), (x,)).exported_program().graph_module

        p = ReplaceSelectWithViewOpPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

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
        class Transpose(torch.nn.Module):
            def forward(self, x):
                return x.transpose(dim0, dim1)

        _max_value = 127
        x = (torch.rand(shape) * _max_value).to(dtype=dtype)
        graph_module = export_to_edge(Transpose(), (x,)).exported_program().graph_module

        p = ReplaceNopTransposeOrPermuteWithViewPass()

        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

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
            [(3, 1, 3, 1, 4), (0, 2, 4, 1, 3), torch.float32],
            [(1, 3, 4), (1, 2, 0), torch.float32],
        ]
    )
    @torch.no_grad()
    def test_replace_nop_permute_with_view(self, input_shape, dims, dtype):
        class Permute(torch.nn.Module):
            def forward(self, x):
                return torch.permute(x, dims)

        x = torch.randn(input_shape).to(dtype=dtype)
        graph_module = export_to_edge(Permute(), (x,)).exported_program().graph_module

        p = ReplaceNopTransposeOrPermuteWithViewPass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

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
            [(3, 4), [1, 0], torch.float32],
            [(3, 4, 6), (0, 2, 1), torch.float32],
        ]
    )
    @torch.no_grad()
    def test_replace_permute_with_transpose(self, input_shape, dims, dtype):
        class Permute(torch.nn.Module):
            def forward(self, x):
                return torch.permute(x, dims)

        x = torch.randn(input_shape).to(dtype=dtype)
        graph_module = export_to_edge(Permute(), (x,)).exported_program().graph_module

        p = ReplacePermuteWithTransposePass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        # Assert that permute op was replaced by a transpose op
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int), 1
        )

    @parameterized.expand(
        [
            # permutations replaced by transpose
            [(3, 4), [0, 1], torch.float32],
        ]
    )
    @torch.no_grad()
    def test_replace_permute_with_transpose_nop(self, input_shape, dims, dtype):
        class Permute(torch.nn.Module):
            def forward(self, x):
                return torch.permute(x, dims)

        x = torch.randn(input_shape).to(dtype=dtype)
        graph_module = export_to_edge(Permute(), (x,)).exported_program().graph_module

        p = ReplacePermuteWithTransposePass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        # Assert that permute op was replaced by a transpose op
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.permute_copy.default), 0
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.transpose_copy.int), 0
        )

    def test_replace_aten_linalg_vector_norm_with_cadence_linalg_vector_norm(self):
        class LinalgVectorNorm(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.linalg.vector_norm(x)

        x = torch.randn(32)

        graph_module = (
            export_to_edge(LinalgVectorNorm(), (x,)).exported_program().graph_module
        )

        p = ReplaceAtenLinalgVectorNormWithCadenceLinalgVectorNormPass()
        graph_after_passes = cast(PassResult, p(graph_module)).graph_module

        # Assert that aten.linalg_vector_norm op was replaced by a
        # cadence.linalg_vector_norm op
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.linalg_vector_norm.default,
            ),
            0,
        )
        self.assertEqual(
            count_node(
                graph_after_passes, exir_ops.edge.cadence.linalg_vector_norm.default
            ),
            1,
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
