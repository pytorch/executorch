# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.decompose_batch_norm import DecomposeBatchNorm
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir import EdgeProgramManager
from executorch.exir.dialects._ops import ops as exir_ops


class TestDecomposeBatchNorm(unittest.TestCase):
    PassStage = RunPasses([DecomposeBatchNorm])
    bn_name = "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default"
    conv_name = "executorch_exir_dialects_edge__ops_aten_convolution_default"

    def setUp(self):
        torch._dynamo.reset()

    class BatchNorm1dNC(torch.nn.Module):
        """Simple BatchNorm1d module with NC input (no spatial dimension)."""

        def __init__(self, num_features: int):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(num_features)
            # Run a forward pass to update the BN running stats.
            self.forward(torch.randn(2, num_features) * 2 + 2)

        def forward(self, x):
            return self.bn(x)

    class BatchNorm1dNCL(torch.nn.Module):
        """Simple BatchNorm1d module with NCL input."""

        def __init__(self, num_features: int):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(num_features)
            # Run a forward pass to update the BN running stats.
            self.forward(torch.randn(2, num_features, 4) * 2 + 2)

        def forward(self, x):
            return self.bn(x)

    class BatchNorm2d(torch.nn.Module):
        """Simple BatchNorm2d module with NCHW input."""

        def __init__(self, num_features: int, affine: bool = True):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(num_features, affine=affine)
            # Run a forward pass to update the BN running stats.
            self.forward(torch.randn(2, num_features, 4, 4) * 2 + 2)

        def forward(self, x):
            return self.bn(x)

    def test_fp32_batch_norm_nc(self):
        """Test that BatchNorm1d with NC input is decomposed to convolution."""
        model = self.BatchNorm1dNC(3).eval()
        tester = (
            Tester(
                model,
                (torch.randn(2, 3),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )
        self._validate_decomposition(tester.get_artifact(), torch.float32, 3, 1)

    def test_fp32_batch_norm_ncl(self):
        """Test that BatchNorm1d with NCL input is decomposed to convolution."""
        model = self.BatchNorm1dNCL(3).eval()
        tester = (
            Tester(
                model,
                (torch.randn(2, 3, 4),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )
        self._validate_decomposition(tester.get_artifact(), torch.float32, 3, 1)

    def test_fp32_batch_norm_nchw(self):
        """Test that BatchNorm2d with NCHW input is decomposed to convolution."""
        model = self.BatchNorm2d(3).eval()
        tester = (
            Tester(
                model,
                (torch.randn(2, 3, 4, 4),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )
        self._validate_decomposition(tester.get_artifact(), torch.float32, 3, 2)

    def test_fp16_batch_norm_nchw(self):
        """Test that BatchNorm2d with NCHW input is decomposed to convolution."""
        model = self.BatchNorm2d(3).to(torch.float16).eval()
        tester = (
            Tester(
                model,
                (torch.randn(2, 3, 4, 4, dtype=torch.float16),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )
        self._validate_decomposition(tester.get_artifact(), torch.float16, 3, 2)

    def test_fp32_batch_norm_nchw_non_affine(self):
        """Test that non-affine BatchNorm2d with NCHW input is decomposed to convolution."""
        model = self.BatchNorm2d(3, affine=False).eval()
        tester = (
            Tester(
                model,
                (torch.randn(2, 3, 4, 4),),
            )
            .export()
            .to_edge()
            .check_count({self.bn_name: 1})
            .run_passes(self.PassStage)
            .check_count({self.conv_name: 1})
            .check_not([self.bn_name])
            .run_method_and_compare_outputs()
        )
        self._validate_decomposition(tester.get_artifact(), torch.float32, 3, 2)

    def _validate_decomposition(
        self,
        edge_manager: EdgeProgramManager,
        dtype: torch.dtype,
        num_channels: int,
        spatial_dims: int,
    ):
        # Verify that the graph contains a 1x1 depthwise convolution and that
        # the transformed parameter dtypes match the original.

        conv_node = next(
            n
            for n in edge_manager.exported_program().graph.nodes
            if n.target == exir_ops.edge.aten.convolution.default
        )
        self.assertEqual(conv_node.meta["val"].dtype, dtype)

        self.assertEqual(len(conv_node.args), 9)
        (
            _,
            w_node,
            b_node,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = conv_node.args

        # Check the convolution parameters. It should be 1x1 depthwise convolution.
        self.assertEqual(stride, [1] * spatial_dims)
        self.assertEqual(padding, [0] * spatial_dims)
        self.assertEqual(dilation, [1] * spatial_dims)
        self.assertEqual(transposed, False)
        self.assertEqual(output_padding, [0] * spatial_dims)
        self.assertEqual(groups, num_channels)

        w_meta = w_node.meta["val"]
        b_meta = b_node.meta["val"]

        # Weight should be (out_c, in_c/g, kH, [kW])
        # Bias should be (out_c)
        self.assertEqual(w_meta.shape, tuple([num_channels, 1] + [1] * spatial_dims))
        self.assertEqual(w_meta.dtype, dtype)
        self.assertEqual(b_meta.shape, (num_channels,))
        self.assertEqual(b_meta.dtype, dtype)
