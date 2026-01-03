# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit Tests for PointwiseAffineRewritePass.

Test coverage:
1. Positive cases: verify pass matches and rewrites patterns (conv2d/mm exists AFTER pass runs)
2. Negative cases: verify pass does NOT match invalid patterns (neither conv2d nor mm created)
3. Coverage gaps: ambiguous channel axis, computed bias, non-layout op interruption
4. Integration: verify end-to-end with XNNPACK partitioner

These tests check graph nodes directly using node.target comparison, not string
matching on graph_module.code. Exception: integration tests use string matching
for "lowered_module" to verify XNNPACK delegation, since module attribute names
are not available as node.target.
"""

import unittest

import torch
import torch.nn as nn

from executorch.backends.test.harness.stages.stage import StageType
from executorch.backends.xnnpack._passes import XNNPACKRemoveCloneOpsTransform
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack._passes.pointwise_affine_pass import (
    ACTIVATION_OPS,
    PointwiseAffineRewritePass,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.const_prop_pass import ConstPropPass
from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass


# Op targets for direct node.target comparison (NOT string matching)
# Use edge ops since PointwiseAffineRewritePass creates edge ops for FuseActivationPass compatibility
CONV_OP = exir_ops.edge.aten.convolution.default
MM_OP = exir_ops.edge.aten.mm.default
ADD_OP = exir_ops.edge.aten.add.Tensor
RELU_OP = exir_ops.edge.aten.relu.default
LINEAR_OP = torch.ops.aten.linear.default
ADDMM_OP = torch.ops.aten.addmm.default


class TestMatcherPositive(unittest.TestCase):
    """Positive tests: verify pass matches and rewrites patterns.

    These tests run prerequisite passes + PointwiseAffineRewritePass and verify
    that conv2d/mm is created in the graph using direct node.target comparison.
    """

    # Prerequisite passes in order, up to and including PointwiseAffineRewritePass
    # This matches XNNPACKPassManager order
    PassStage = RunPasses(
        [
            XNNPACKRemoveCloneOpsTransform,
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ConstPropPass,
            PointwiseAffineRewritePass,
        ]
    )

    def setUp(self):
        torch.manual_seed(42)
        torch._dynamo.reset()

    def test_nchw_permute_flatten_linear_unflatten_permute(self):
        """Test NCHW pattern is matched and rewritten to Conv2d."""

        class NCHWPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout)
                x = x.permute(0, 3, 1, 2)
                return x

        model = NCHWPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # Run passes and verify conv2d was created using direct node.target check
        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            # Verify exactly 1 conv2d was created
            .check_node_count({CONV_OP: 1})
            # Verify no linear ops remain
            .check_node_count({LINEAR_OP: 0, ADDMM_OP: 0})
            .run_method_and_compare_outputs()
        )

    def test_nhwc_flatten_matmul_reshape(self):
        """Test NHWC pattern is matched and rewritten to mm."""

        class NHWCPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, h, w, c = x.shape
                x = x.reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout)
                return x

        model = NHWCPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 4, 4, 8),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({MM_OP: 1})  # mm exists - pass worked!
            .check_node_count({LINEAR_OP: 0, ADDMM_OP: 0})
            .run_method_and_compare_outputs()
        )

    def test_rank3_transformer_linear(self):
        """Test rank=3 pattern [B, T, C] is matched and rewritten to mm."""

        class TransformerLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, t, c = x.shape
                x = x.reshape(b * t, c)
                x = self.linear(x)
                x = x.reshape(b, t, self.cout)
                return x

        model = TransformerLinear(64, 128).eval()
        inputs = (torch.randn(2, 16, 64),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({MM_OP: 1})  # mm exists - pass worked!
            .check_node_count({LINEAR_OP: 0, ADDMM_OP: 0})
            .run_method_and_compare_outputs()
        )

    def test_bias_as_separate_add(self):
        """Test pattern with bias as a separate Add is matched."""

        class PointwiseLinearSeparateBias(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.weight = nn.Parameter(torch.randn(cout, cin))
                self.bias = nn.Parameter(torch.randn(cout))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = torch.mm(x, self.weight.t())
                x = x + self.bias
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = PointwiseLinearSeparateBias(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 1})  # Conv2d exists - pass worked!
            .run_method_and_compare_outputs()
        )

    def test_bias_broadcast_shape(self):
        """Test pattern with broadcast bias shape (1, cout) is matched."""

        class PointwiseLinearBroadcastBias(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.weight = nn.Parameter(torch.randn(cout, cin))
                self.bias = nn.Parameter(torch.randn(1, cout))  # (1, cout) broadcast

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = torch.mm(x, self.weight.t())
                x = x + self.bias  # broadcast add
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = PointwiseLinearBroadcastBias(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 1})  # Conv2d exists - pass worked!
            .run_method_and_compare_outputs()
        )

    def test_linear_activation_layout_matches(self):
        """Test linear -> relu -> layout is matched and rewritten with relu preserved."""

        class LinearReluLayout(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                x = torch.relu(x)  # Activation before layout ops
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = LinearReluLayout(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 1})  # Conv2d exists - pass worked!
            .check_node_count({RELU_OP: 1})  # ReLU preserved
            .check_node_count({LINEAR_OP: 0, ADDMM_OP: 0})
            .run_method_and_compare_outputs()
        )

    def test_linear_bias_add_activation_layout_matches(self):
        """Test linear -> add(bias) -> relu -> layout is matched with both preserved."""

        class LinearBiasReluLayout(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.weight = nn.Parameter(torch.randn(cout, cin))
                self.bias = nn.Parameter(torch.randn(cout))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = torch.mm(x, self.weight.t())
                x = x + self.bias  # Bias add
                x = torch.relu(x)  # Activation after bias
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = LinearBiasReluLayout(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 1})  # Conv2d exists - pass worked!
            .check_node_count({RELU_OP: 1})  # ReLU preserved
            .run_method_and_compare_outputs()
        )


class TestMatcherNegative(unittest.TestCase):
    """Negative tests: verify pass does NOT match invalid patterns.

    These tests verify that patterns that should NOT be rewritten
    (spatial mixing, broken output restore, etc.) are left unchanged.
    IMPORTANT: We check that NEITHER conv2d NOR mm was created using node.target.
    """

    # Same pass stage as positive tests
    PassStage = RunPasses(
        [
            XNNPACKRemoveCloneOpsTransform,
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ConstPropPass,
            PointwiseAffineRewritePass,
        ]
    )

    def setUp(self):
        torch.manual_seed(42)
        torch._dynamo.reset()

    def test_spatial_mixing_should_not_match(self):
        """Linear that mixes H*W with C should NOT match (not pointwise)."""

        class SpatialMixingLinear(nn.Module):
            def __init__(self, cin: int, h: int, w: int, cout: int):
                super().__init__()
                self.linear = nn.Linear(cin * h * w, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.reshape(n, c * h * w)
                x = self.linear(x)
                return x

        model = SpatialMixingLinear(8, 4, 4, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # Run passes, verify NEITHER conv2d NOR mm was created
        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # No rewrite happened
            .run_method_and_compare_outputs()
        )

    def test_output_shape_mismatch_should_not_match(self):
        """Output reshape that doesn't restore proper shape should NOT match."""

        class BrokenOutputRestore(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                # BROKEN: reshape mixes spatial with channel
                x = x.reshape(n, h, w * self.cout)
                return x

        model = BrokenOutputRestore(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # No rewrite happened
            .run_method_and_compare_outputs()
        )

    def test_multiple_consumers_should_not_match(self):
        """Linear output used by multiple nodes should NOT match."""

        class MultipleConsumers(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                y = self.linear(x)
                # Two uses of linear output
                y1 = y.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                y2 = y.sum()  # Second consumer
                return y1 + y2

        model = MultipleConsumers(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # No rewrite happened
            .run_method_and_compare_outputs()
        )

    def test_residual_add_should_not_match(self):
        """Add with activation (not bias parameter) should NOT match."""

        class ResidualAdd(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout, bias=False)

            def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                # Add residual (activation, not bias) - should stop tracing
                x = x + residual
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = ResidualAdd(8, 16).eval()
        # residual has same shape as linear output
        inputs = (torch.randn(2, 8, 4, 4), torch.randn(32, 16))

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # No rewrite happened
            .run_method_and_compare_outputs()
        )

    def test_ambiguous_channel_axis_should_not_match(self):
        """Pattern where multiple axes could be the channel should NOT match.

        This tests the safety rule: if cin appears in more than one axis and
        the flatten could plausibly preserve either, we must reject.
        """

        class AmbiguousChannelAxis(nn.Module):
            def __init__(self):
                super().__init__()
                # Linear with cin=4, and input shape [2, 4, 4] has TWO axes of size 4
                self.linear = nn.Linear(4, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, s1, s2 = x.shape  # [2, 4, 4] - s1 and s2 both have size 4!
                x = x.reshape(b * s1, s2)  # [8, 4] - ambiguous which axis is channel
                x = self.linear(x)  # [8, 8]
                x = x.reshape(b, s1, 8)  # [2, 4, 8]
                return x

        model = AmbiguousChannelAxis().eval()
        # Input [2, 4, 4] has two axes with size 4 - which is the channel?
        inputs = (torch.randn(2, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # Should reject ambiguity
            .run_method_and_compare_outputs()
        )

    def test_computed_bias_should_not_match(self):
        """Bias computed from activation (not a parameter) should NOT match.

        This tests that we don't accidentally treat a computed tensor
        (like mean, reduce) as a bias just because it's broadcastable.
        """

        class ComputedBias(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                y = self.linear(x)
                # Bias is COMPUTED from the output (not a parameter)
                bias = y.mean(dim=0)  # Shape [cout], broadcastable
                y = y + bias  # Looks like bias add but isn't a parameter
                y = y.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return y

        model = ComputedBias(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # Should not match
            .run_method_and_compare_outputs()
        )

    def test_non_layout_op_interrupts_pattern(self):
        """Non-layout op between linear and output restore should NOT match.

        This tests that we correctly stop tracing when encountering an
        operation that isn't a layout op or allowed activation (permute, reshape, relu, etc.).
        Tanh is not in ACTIVATION_OPS, so it should interrupt the pattern.
        """
        # Verify test assumption: tanh is not in ACTIVATION_OPS
        # If this assertion fails, the test needs to use a different op
        self.assertTrue(
            all(torch.ops.aten.tanh.default != op for op in ACTIVATION_OPS),
            "Test assumption violated: tanh is now in ACTIVATION_OPS. "
            "Update this test to use a different non-activation op.",
        )

        class NonLayoutOpInterruption(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                # Tanh interrupts the pattern - not an allowed activation
                x = torch.tanh(x)
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = NonLayoutOpInterruption(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # Should stop at tanh
            .run_method_and_compare_outputs()
        )

    def test_two_activations_should_not_match(self):
        """Two activations in a row should NOT match (only one allowed).

        The pass only accepts at most one activation op in the canonical position.
        Having relu -> gelu should interrupt the pattern.
        """

        class TwoActivations(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                x = torch.relu(x)  # First activation
                x = torch.nn.functional.gelu(
                    x
                )  # Second activation - should break pattern
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = TwoActivations(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count(
                {CONV_OP: 0, MM_OP: 0}
            )  # Should not match due to two activations
            .run_method_and_compare_outputs()
        )

    def test_channel_axis_not_at_last_position_should_not_match(self):
        """Reshape without permute when channel is not last should NOT match.

        Reviewer's example: x: [1, 2, 3, 4] -> reshape(12, 2) -> linear -> reshape(1, 4, 3, 4)

        This tests the core validation logic: reshaping [1, 2, 3, 4] -> [12, 2]
        puts axis 1 (size=2) into the last position of the 2D tensor, but there
        was NO permute to move it there. The pass should reject this because:
        1. cin=2 is found at axis 1 (not the last position in [1, 2, 3, 4])
        2. Producer of the reshape is NOT a permute op

        If this pattern were accepted, the pass would incorrectly treat it as
        pointwise, but the reshape actually merges spatial dims with channel.
        The output [1, 4, 3, 4] has channel at axis 1 (size 4), not preserving
        the original spatial structure.
        """

        class ChannelNotLastReshape(nn.Module):
            def __init__(self):
                super().__init__()
                # Input: [1, 2, 3, 4] -> flat: [12, 2] -> linear(2, 4) -> [12, 4] -> [1, 4, 3, 4]
                self.linear = nn.Linear(2, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [1, 2, 3, 4]
                # This reshape merges batch + channel + spatial dimensions together
                x = x.reshape(12, 2)  # NO permute! Channel axis not preserved
                x = self.linear(x)  # [12, 4]
                x = x.reshape(1, 4, 3, 4)  # Reshape output with new dimensions
                return x

        model = ChannelNotLastReshape().eval()
        inputs = (torch.randn(1, 2, 3, 4),)

        # This should NOT be rewritten because the channel axis (axis 1 with size 2)
        # is not at the last position and there's no permute to move it there.
        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # Should NOT match
            .run_method_and_compare_outputs()
        )


class TestNumericalEquivalence(unittest.TestCase):
    """Verify transformed program produces same output as original.

    This is implicitly tested by run_method_and_compare_outputs() in other tests,
    but this class provides an explicit, focused numerical correctness check.
    """

    PassStage = RunPasses(
        [
            XNNPACKRemoveCloneOpsTransform,
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ConstPropPass,
            PointwiseAffineRewritePass,
        ]
    )

    def setUp(self):
        torch.manual_seed(42)
        torch._dynamo.reset()

    def test_nchw_output_matches_original(self):
        """Verify NCHW rewrite produces numerically equivalent output."""

        class NCHWPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout)
                x = x.permute(0, 3, 1, 2)
                return x

        model = NCHWPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # run_method_and_compare_outputs() compares original vs transformed
        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .run_method_and_compare_outputs()
        )


class TestXNNPACKIntegration(unittest.TestCase):
    """Integration tests with XNNPACK partitioner.

    Verifies that the rewritten Conv2d/mm ops are properly delegated to XNNPACK.
    Uses the full to_edge_transform_and_lower() pipeline.
    """

    # Same pass stage for pre-partition verification
    PassStage = RunPasses(
        [
            XNNPACKRemoveCloneOpsTransform,
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ConstPropPass,
            PointwiseAffineRewritePass,
        ]
    )

    def setUp(self):
        torch.manual_seed(42)
        torch._dynamo.reset()

    def test_nchw_delegates_to_xnnpack(self):
        """Verify NCHW Conv2d gets delegated to XNNPACK.

        This test verifies both:
        1. The rewrite occurred (conv2d in pre-partition graph)
        2. The delegation occurred (lowered_module in final graph)
        """

        class NCHWPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout)
                x = x.permute(0, 3, 1, 2)
                return x

        model = NCHWPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # First verify the rewrite occurred with run_passes using node.target
        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 1})  # Verify rewrite before partitioning
        )

        # Then verify full pipeline delegates to XNNPACK
        # Note: check() with string is fine here for "lowered_module" since it's
        # a module attribute name, not an op target
        (
            Tester(model, inputs)
            .export()
            .to_edge_transform_and_lower()
            .check(["lowered_module"])  # Delegated to XNNPACK
            .run_method_and_compare_outputs()
        )

    def test_nhwc_delegates_to_xnnpack(self):
        """Verify NHWC mm gets delegated to XNNPACK.

        For NHWC (channel-last) patterns, the pass rewrites linear -> mm.
        The intent is to convert from the generic linear op to an explicit
        mm (matrix multiply) which XNNPACK can delegate and accelerate.

        Unlike NCHW which becomes Conv2d(1x1), NHWC stays as mm because:
        - Channel is already at the last position (no permute needed)
        - mm is directly supported by XNNPACK
        - The rewrite still eliminates the linear op's implicit weight transpose

        This test verifies:
        1. The rewrite occurred (mm in pre-partition graph)
        2. The delegation occurred (lowered_module in final graph)
        """

        class NHWCPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, h, w, c = x.shape
                x = x.reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout)
                return x

        model = NHWCPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 4, 4, 8),)

        # First verify the rewrite occurred: linear -> mm
        (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({MM_OP: 1})  # Verify mm was created
            .check_node_count({LINEAR_OP: 0, ADDMM_OP: 0})  # No linear ops remain
        )

        # Then verify full pipeline delegates to XNNPACK
        (
            Tester(model, inputs)
            .export()
            .to_edge_transform_and_lower()
            .check(["lowered_module"])  # Delegated to XNNPACK
            .run_method_and_compare_outputs()
        )

    def test_original_weights_not_in_transformed_graph(self):
        """Verify original linear weights are removed after transformation.

        When the pass rewrites linear -> conv2d/mm, it creates NEW weight tensors
        (reshaped for conv2d or transposed for mm). The original linear weight
        placeholders should be erased from the graph.

        This test verifies:
        1. Before pass: graph has placeholders for 'linear.weight' and 'linear.bias'
        2. After pass: those placeholders are removed, replaced by new conv weights
        """

        class NCHWPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1)
                x = x.reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout)
                x = x.permute(0, 3, 1, 2)
                return x

        model = NCHWPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # Run the pass and get the artifact
        tester = Tester(model, inputs).export().to_edge().run_passes(self.PassStage)

        # Get the exported program after passes
        ep = tester.get_artifact(StageType.RUN_PASSES).exported_program()
        gm = ep.graph_module

        # Collect all placeholder names
        placeholder_names = [
            node.name for node in gm.graph.nodes if node.op == "placeholder"
        ]

        # Original linear weight/bias names should NOT be present
        # They should have been erased and replaced with conv weights
        for name in placeholder_names:
            self.assertNotIn(
                "linear_weight",
                name,
                f"Original linear weight placeholder '{name}' should have been removed",
            )
            self.assertNotIn(
                "linear_bias",
                name,
                f"Original linear bias placeholder '{name}' should have been removed",
            )

        # New conv weights should be present (contains 'conv_w' or 'conv_b')
        conv_weight_found = any("conv_w" in name for name in placeholder_names)
        self.assertTrue(
            conv_weight_found,
            f"Expected conv weight placeholder, found: {placeholder_names}",
        )


class TestFuseActivationPassIntegration(unittest.TestCase):
    """Integration tests with FuseActivationPass.

    Verifies that the PointwiseAffineRewritePass creates proper edge ops
    that can be fused by FuseActivationPass (conv2d + relu -> fused conv2d).

    Note: FuseActivationPass embeds activation constraints as metadata and removes
    the activation node. This metadata is only consumed by the XNNPACK serializer,
    not by the Python graph module execution. So we verify fusion occurred via
    metadata inspection, not output comparison.
    """

    # Pass stage that includes PointwiseAffineRewritePass THEN FuseActivationPass
    PassStageWithFusion = RunPasses(
        [
            XNNPACKRemoveCloneOpsTransform,
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ConstPropPass,
            PointwiseAffineRewritePass,
            FuseActivationPass,  # Should fuse relu into conv2d
        ]
    )

    def setUp(self):
        torch.manual_seed(42)
        torch._dynamo.reset()

    def test_conv2d_relu_fusion(self):
        """Verify conv2d + relu from PointwiseAffineRewritePass gets fused.

        After PointwiseAffineRewritePass: conv2d -> relu
        After FuseActivationPass: conv2d (with fused activation metadata), no relu node
        """

        class LinearReluLayout(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                x = torch.relu(x)  # Activation before layout ops
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = LinearReluLayout(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # After both passes: conv2d exists, relu is fused (removed)
        tester = (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStageWithFusion)
            .check_node_count({CONV_OP: 1})  # Conv2d exists
            .check_node_count({RELU_OP: 0})  # ReLU was fused (removed)
            # Note: Don't call run_method_and_compare_outputs() because FuseActivationPass
            # embeds activation as metadata, which is only consumed by XNNPACK serializer
        )

        # Verify the fused activation metadata is set on the conv2d node
        gm = tester.get_artifact(StageType.RUN_PASSES).exported_program().graph_module
        for node in gm.graph.nodes:
            if node.target == CONV_OP:
                fused_tag = node.meta.get(FuseActivationPass.FUSED_ACTIVATION_TAG)
                self.assertIsNotNone(
                    fused_tag,
                    "Conv2d should have fused activation metadata after FuseActivationPass",
                )
                # ReLU has output_min=0, output_max=+inf
                self.assertEqual(fused_tag.output_min, 0)
                break
        else:
            self.fail("No conv2d node found in graph")

    def test_conv2d_bias_relu_fusion(self):
        """Verify conv2d (with bias) + relu pattern gets fused correctly.

        Tests: linear -> add(bias) -> relu -> layout
        After PointwiseAffineRewritePass: conv2d(with bias) -> relu
        After FuseActivationPass: conv2d(with bias and fused activation)
        """

        class LinearBiasReluLayout(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.weight = nn.Parameter(torch.randn(cout, cin))
                self.bias = nn.Parameter(torch.randn(cout))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = torch.mm(x, self.weight.t())
                x = x + self.bias  # Bias add
                x = torch.relu(x)  # Activation after bias
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = LinearBiasReluLayout(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        tester = (
            Tester(model, inputs)
            .export()
            .to_edge()
            .run_passes(self.PassStageWithFusion)
            .check_node_count({CONV_OP: 1})  # Conv2d with bias
            .check_node_count({RELU_OP: 0})  # ReLU was fused
        )

        # Verify the fused activation metadata is set
        gm = tester.get_artifact(StageType.RUN_PASSES).exported_program().graph_module
        conv_found = False
        for node in gm.graph.nodes:
            if node.target == CONV_OP:
                conv_found = True
                fused_tag = node.meta.get(FuseActivationPass.FUSED_ACTIVATION_TAG)
                self.assertIsNotNone(
                    fused_tag,
                    "Conv2d should have fused activation metadata after FuseActivationPass",
                )
                self.assertEqual(fused_tag.output_min, 0)
                break
        self.assertTrue(conv_found, "No conv2d node found in graph")

    def test_full_pipeline_with_xnnpack_delegation(self):
        """Verify full pipeline: rewrite + fusion + XNNPACK delegation.

        This test uses to_edge_transform_and_lower() which runs the full XNNPACK
        pass pipeline including FuseActivationPass, then delegates to XNNPACK.
        The delegation path properly handles fused activation metadata.
        """

        class LinearReluLayout(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                x = torch.relu(x)
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = LinearReluLayout(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # Full pipeline with XNNPACK delegation - this properly handles fused activation
        (
            Tester(model, inputs)
            .export()
            .to_edge_transform_and_lower()
            .check(["lowered_module"])  # Delegated to XNNPACK
            .run_method_and_compare_outputs()  # XNNPACK runtime handles fused activation
        )


class TestQuantizationCompatibility(unittest.TestCase):
    """Tests to verify pass does NOT match quantized graphs.

    Quantized graphs have Q/DQ nodes around the linear op, which changes the
    graph structure. The pass should NOT match these patterns because:
    1. The weight node is a DQ output, not a parameter placeholder
    2. The linear input comes from a DQ node, not a layout op

    These tests explicitly verify that the pass does not introduce conv2d/mm
    when run on quantized graphs, and that the full pipeline still works.
    """

    # Pass stage for explicit matching verification
    PassStage = RunPasses(
        [
            XNNPACKRemoveCloneOpsTransform,
            DimOrderOpsRevertPass,
            ConvertToLinearPass,
            ConstPropPass,
            PointwiseAffineRewritePass,
        ]
    )

    def setUp(self):
        torch.manual_seed(42)
        torch._dynamo.reset()

    def test_quantized_nchw_linear_not_matched(self):
        """Verify PointwiseAffineRewritePass does NOT match quantized NCHW linear.

        The pass should not create conv2d/mm nodes because in a quantized graph,
        the linear's weight comes from a DQ node (not a parameter), and the pass
        requires weights to be parameters.
        """

        class NCHWPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = NCHWPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # Quantize, export, run passes, verify NO conv2d/mm created
        (
            Tester(model, inputs)
            .quantize()  # Creates Q/DQ nodes around linear
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            # Pass should NOT match quantized graph - no conv2d or mm should be created
            .check_node_count({CONV_OP: 0, MM_OP: 0})
        )

    def test_quantized_nhwc_linear_not_matched(self):
        """Verify PointwiseAffineRewritePass does NOT match quantized NHWC linear."""

        class NHWCPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, h, w, c = x.shape
                x = x.reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout)
                return x

        model = NHWCPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 4, 4, 8),)

        (
            Tester(model, inputs)
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_node_count({CONV_OP: 0, MM_OP: 0})  # Should NOT match
        )

    def test_quantized_nchw_linear_full_pipeline(self):
        """Verify quantized NCHW pattern works with full XNNPACK pipeline.

        Even though the pass doesn't match, the full pipeline should still
        successfully delegate to XNNPACK and produce correct outputs.
        """

        class NCHWPointwiseLinear(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = NCHWPointwiseLinear(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        # Full quantized pipeline: quantize -> export -> lower to XNNPACK
        (
            Tester(model, inputs)
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check(["lowered_module"])  # Verify XNNPACK delegation succeeded
            .run_method_and_compare_outputs()  # Verify numerical correctness
        )

    def test_quantized_linear_relu_full_pipeline(self):
        """Verify quantized linear + relu pattern works with full pipeline."""

        class LinearReluLayout(nn.Module):
            def __init__(self, cin: int, cout: int):
                super().__init__()
                self.cout = cout
                self.linear = nn.Linear(cin, cout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n, c, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
                x = self.linear(x)
                x = torch.relu(x)
                x = x.reshape(n, h, w, self.cout).permute(0, 3, 1, 2)
                return x

        model = LinearReluLayout(8, 16).eval()
        inputs = (torch.randn(2, 8, 4, 4),)

        (
            Tester(model, inputs)
            .quantize()
            .export()
            .to_edge_transform_and_lower()
            .check(["lowered_module"])
            .run_method_and_compare_outputs()
        )


if __name__ == "__main__":
    unittest.main()
